import gc
import os
import json

import torch

from e2edet.evaluate import CocoEvaluator
from e2edet.utils.distributed import (
    reduce_dict,
    is_dist_avail_and_initialized,
    get_world_size,
    all_gather,
    synchronize,
    is_master,
)
from e2edet.utils.params import recursive_copy_to_device
from e2edet.dataset.helper import Prefetcher


@torch.no_grad()
def evaluate(split, trainer):
    trainer.writer.write(f"Evaluation time. Running on full {split} set...")
    trainer.timers[split].reset()

    iou_type = trainer.iou_type
    iter_per_update = trainer.iter_per_update
    dataset = trainer.datasets[split]
    dataloader = trainer.dataloaders[split]
    model = trainer.model

    coco_evaluator = None
    accumulated_results = None
    other_args = {}
    if split == "test":
        if trainer.parallel:
            model.module.inference()
        else:
            model.inference()
        accumulated_results = {}
    else:
        model.eval()
        if iou_type is not None:
            coco_evaluator = CocoEvaluator(dataset.coco, iou_type)
            other_args["return_rles"] = split == "test"

    prefetcher = Prefetcher(
        trainer.dataloaders[split], trainer.datasets[split], prefetch=True
    )

    for _ in range(len(dataloader)):
        batch = prefetcher.get_next_sample()

        if iter_per_update > 1:
            results = {}
            for idx, splitted_batch in enumerate(batch):
                outputs, targets = _forward(split, splitted_batch, model, trainer)

                results.update(
                    dataset.format_for_evalai(outputs, targets, **other_args)
                )
        else:
            outputs, targets = _forward(split, batch, model, trainer)

            results = dataset.format_for_evalai(outputs, targets, **other_args)
        trainer.profile("Post-processing time")

        if coco_evaluator is not None:
            coco_evaluator.update(results)

        if accumulated_results is not None:
            accumulated_results.update(
                recursive_copy_to_device(results, True, torch.device("cpu"))
            )

    stats = {
        "update": trainer.current_update,
        "epoch": trainer.current_epoch,
        "max_update": trainer.max_update,
        "num_image": len(dataset),
    }
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if "bbox" in iou_type:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_type:
            stats["coco_eval_segm"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if accumulated_results is not None:
        accumulated_results = all_gather(accumulated_results)

        if is_master():
            merged_results = {}
            for result in accumulated_results:
                merged_results.update(result)
            accumulated_results = None

            if iou_type is None:
                save_file = os.path.join(
                    trainer.checkpoint.ckpt_foldername, "results.pth"
                )
                torch.save(merged_results, save_file)

                dataset.prepare_for_evaluation(
                    merged_results, trainer.checkpoint.ckpt_foldername
                )
            else:
                merged_results = dataset.prepare_for_evaluation(merged_results)
                test_path = os.path.join(
                    trainer.checkpoint.ckpt_foldername, "test_result.json"
                )
                json.dump(merged_results, open(test_path, "w"))
        synchronize()

    trainer._print_log(split, stats)
    trainer._update_tensorboard(split)

    model.train()
    gc.collect()
    if "cuda" in str(trainer.device):
        torch.cuda.empty_cache()

    trainer.timers["train"].reset()


def train_epoch(trained_batch_idx, trainer):
    current_epoch = trainer.current_epoch
    current_update = trainer.current_update
    max_update = trainer.max_update
    iter_per_update = trainer.iter_per_update
    max_norm = trainer.running_config.max_norm
    eval_interval = trainer.eval_interval
    save_interval = trainer.save_interval

    model = trainer.model
    optimizer = trainer.optimizer
    lr_scheduler = trainer.lr_scheduler

    prefetcher = Prefetcher(
        trainer.dataloaders["train"], trainer.datasets["train"], prefetch=True
    )

    if trainer.samplers["train"] is not None:
        trainer.samplers["train"].set_epoch(current_epoch)

    for idx in range(len(trainer.dataloaders["train"])):
        # for idx, batch in enumerate(trainer.dataloaders["train"]):
        batch = prefetcher.get_next_sample()
        if idx < trained_batch_idx:
            continue

        optimizer.zero_grad()
        if iter_per_update > 1:
            num_boxes = 0
            for split in batch:
                num_boxes += sum(len(t["labels"]) for t in split[1])
            num_boxes = torch.tensor(
                [num_boxes], dtype=torch.float, device=trainer.device
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)

            for idx, splitted_batch in enumerate(batch):
                splitted_batch[0]["num_boxes"] = num_boxes
                finished_batch = (idx + 1) == len(batch)
                output, _ = _forward("train", splitted_batch, model, trainer)
                _update_info("train", output, trainer, finished_batch)
                _backward(output, trainer)
        else:
            output, _ = _forward("train", batch, model, trainer)
            _update_info("train", output, trainer)
            _backward(output, trainer)
        current_update = _step(optimizer, max_norm, current_update, trainer)

        if current_update == trainer.current_update:
            continue

        assert trainer.current_update == (current_update - 1)
        trainer.current_update = current_update

        lr_scheduler.step(current_update)

        if current_update % save_interval == 0:
            trainer.writer.write("Checkpoint time. Saving a checkpoint...")
            trainer.checkpoint.save(current_update)

        if current_update % eval_interval == 0 and "val" in trainer.run_type:
            evaluate("val", trainer)

        if current_update > max_update:
            break

    lr_scheduler.step_epoch(current_epoch)


def _forward(split, batch, model, trainer):
    trainer.profile("Batch prepare time")

    sample, target = batch

    if trainer.use_fp16 and split == "train":
        assert trainer.use_fp16 in ("float16", "bfloat16")
        dtype = torch.bfloat16 if trainer.use_fp16 == "bfloat16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = model(sample, target)
    else:
        output = model(sample, target)
    trainer.profile("Forward time")

    return output, target


def _backward(output, trainer):
    loss = output["losses"]

    if trainer.use_fp16:
        trainer.grad_scaler.scale(loss).backward()
    else:
        loss.backward()
    trainer.profile("Backward time")


def _step(optimizer, max_norm, current_update, trainer):
    if trainer.use_fp16:
        trainer.grad_scaler.unscale_(optimizer)

    norm = optimizer.clip_grad_norm(max_norm)
    if trainer.tb_writer is not None:
        trainer.tb_writer.add_scalars({"total_norm": norm}, current_update)

    if trainer.use_fp16:
        trainer.grad_scaler.step(optimizer)
        trainer.grad_scaler.update()
    else:
        optimizer.step()

    if torch.isnan(norm).item() or torch.isinf(norm).item():
        return current_update
    current_update += 1

    return current_update


@torch.no_grad()
def _update_info(split, output, trainer, finished_batch=True):
    current_update = trainer.current_update
    log_interval = trainer.log_interval

    update_dict = _sync_losses_and_metrics(output)
    batch_size = trainer.running_config.batch_size
    trainer.meters[split].update(update_dict, batch_size)

    if (
        split == "train"
        and (current_update % log_interval == 0)
        and current_update > 0
        and finished_batch
    ):
        stats = {}
        ups = log_interval / trainer.timers["train"].unix_time_since_start()
        if "cuda" in str(trainer.device):
            stats["max mem"] = torch.cuda.max_memory_allocated() / 1000
            stats["max mem"] //= 1000

        total_update = len(trainer.dataloaders["train"])
        num_epoch = (current_update + total_update - 1) // total_update

        stats.update(
            {
                "epoch": num_epoch,
                "update": trainer.current_update,
                "max_update": trainer.max_update,
                "lr": [
                    param_group["lr"] for param_group in trainer.optimizer.param_groups
                ],
                "ups": "{:.2f}".format(ups),
                "time": trainer.timers["train"].get_time_since_start(),
                "time_since_start": trainer.total_timer.get_time_since_start(),
                "eta": trainer._calculate_time_left(),
            }
        )
        trainer._print_log(split, stats)
        trainer._update_tensorboard(split)
        trainer.timers["train"].reset()


def _sync_losses_and_metrics(output):
    losses = output["losses_stat"]
    metrics = output["metrics"]

    reduced_losses = reduce_dict(losses)
    reduced_metrics = reduce_dict(metrics)

    update_dict = {}
    update_dict.update(reduced_losses)
    update_dict.update(reduced_metrics)

    return update_dict

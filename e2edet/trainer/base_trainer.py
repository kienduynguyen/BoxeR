# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import time
import os
import collections

import torch
from torch.cuda.amp import GradScaler

from e2edet.utils.meter import Meter
from e2edet.utils.checkpoint import Checkpoint
from e2edet.utils.distributed import (
    is_master,
    is_dist_avail_and_initialized,
    synchronize,
)
from e2edet.utils.general import print_model_parameters
from e2edet.utils.logger import Logger, TensorboardLogger
from e2edet.utils.timer import Timer
from e2edet.model import build_model
from e2edet.optim import build_optimizer
from e2edet.optim.scheduler import build_scheduler
from e2edet.dataset import build_dataset, build_dataloader
from e2edet.trainer import register_trainer
from e2edet.trainer.engine import evaluate, train_epoch


@register_trainer("base_trainer")
class BaseTrainer:
    def __init__(self, configuration):
        self.configuration = configuration
        self.config = self.configuration.get_config()
        self.profiler = Timer()
        self.total_timer = Timer()
        self.running_config = self.config.training
        if self.configuration is not None:
            self.args = self.configuration.args

    def load(self):
        self._set_device()
        self.run_type = self.running_config.get("run_type", "train")
        self.writer = Logger(
            self.running_config.save_dir,
            self.running_config.logger_level,
            self.running_config.log_format,
            self.running_config.should_not_log,
        )
        self.configuration.pretty_print(self.writer)
        self.config_based_setup()
        self.load_task()
        self.load_model_and_optimizer()

    def _set_device(self):
        self.local_rank = self.config.device_id

        if self.config.distributed.init_method is not None:
            self.device = torch.device("cuda", self.local_rank)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def config_based_setup(self):
        seed = self.running_config.seed
        if seed is None:
            # guard against training without seed
            raise ValueError("seed should not be None")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    def load_task(self):
        self.writer.write("Loading datasets", "info")

        self.datasets = {}
        self.dataloaders = {}
        self.samplers = {}
        self._load_split_task(["train", "val", "test"])

    def _load_split_task(self, splits):
        self.splits = []
        for split in splits:
            dataset, dataloader, sampler = None, None, None
            if split in self.run_type:
                dataset = build_dataset(self.config, split, self.device)
                dataloader, sampler = build_dataloader(self.config, split, dataset)
                self.splits.append(split)

            self.datasets[split] = dataset
            self.dataloaders[split] = dataloader
            self.samplers[split] = sampler

    def load_model_and_optimizer(self):
        self.writer.write("Loading model and optimizer", "info")

        num_classes = self.datasets[self.splits[0]].get_answer_size()
        self.model = build_model(self.config, num_classes)

        if "cuda" in str(self.device):
            device_info = "CUDA Device {} is: {}".format(
                self.config.distributed.rank,
                torch.cuda.get_device_name(self.local_rank),
            )
            self.writer.write(device_info, log_all=True)

        self.model = self.model.to(self.device)
        self.optimizer = build_optimizer(self.config, self.model)
        self.lr_scheduler = build_scheduler(self.config, self.optimizer)

        self._parallelize_model()
        self._init_params_and_checkpoint()

    def _parallelize_model(self):
        self.parallel = False
        if "cuda" in str(self.device) and is_dist_avail_and_initialized():
            find_unused_parameters = self.running_config.find_unused_parameters
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=find_unused_parameters,
            )
            self.parallel = True

    def _init_params_and_checkpoint(self):
        self.writer.write("Torch version is: " + torch.__version__)

        self.use_fp16 = (
            False
            if self.running_config.use_fp16 == "none"
            else self.running_config.use_fp16
        )
        self.grad_scaler = GradScaler() if self.use_fp16 else None

        self.checkpoint = Checkpoint(self)
        self.log_interval = self.running_config.log_interval
        self.eval_interval = self.running_config.evaluation_interval
        self.save_interval = self.running_config.checkpoint_interval
        self.iter_per_update = self.running_config.iter_per_update
        self.iou_type = (
            tuple(self.running_config.iou_type)
            if self.running_config.iou_type is not None
            else None
        )

        self.max_update = self.running_config.max_update
        self.max_epoch = self.running_config.max_epoch

        if self.max_epoch is not None and self.max_update is not None:
            raise ValueError("max_epoch and max_update are mutually exclusive!")

        if self.dataloaders["train"] is not None:
            if self.max_epoch is not None:
                self.max_update = self.max_epoch * len(self.dataloaders["train"])
            self.eval_interval = int(
                self.eval_interval * len(self.dataloaders["train"])
            )
            self.save_interval = int(
                self.save_interval * len(self.dataloaders["train"])
            )
        else:
            self.max_update = 0

        self.meters = {
            "train": Meter(window_size=self.log_interval * self.iter_per_update),
            "val": Meter(),
            "test": Meter(),
        }
        self.timers = {"train": Timer(), "val": Timer(), "test": Timer()}

        self.eval_iteration = 0
        if self.dataloaders["val"] is not None:
            self.eval_iteration = len(self.dataloaders["val"])

        self.current_epoch = 0
        self.current_update = 0

        self.is_resumed = self.checkpoint.load_state_dict()
        self.not_debug = self.running_config.logger_level != "debug"

        self.tb_writer = None
        if self.running_config.tensorboard:
            tb_log_folder = os.path.join(self.writer.log_folder, "tensorboard")

            if self.running_config.tensorboard_logdir:
                tb_log_folder = self.running_config.tensorboard_logdir

            if is_master() and not os.path.exists(tb_log_folder):
                os.makedirs(tb_log_folder)
            synchronize()
            self.tb_writer = TensorboardLogger(tb_log_folder)

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)

        print_model_parameters(self.model, self.writer)

        if "train" not in self.run_type:
            self.inference()
            return

        self.model.train()
        self.profile("Setup Time")

        torch.autograd.set_detect_anomaly(False)
        self.writer.write("Starting training...")

        self._resume_training_state_if_needed()
        while self.current_update < self.max_update:
            self.current_epoch += 1
            train_epoch(0, self)
        self.finalize()

    def _resume_training_state_if_needed(self):
        trained_idx = self.current_update % len(self.dataloaders["train"])
        is_epoch_finished = trained_idx == 0

        if self.is_resumed:
            self.writer.write("Resuming training...")

            if is_epoch_finished:
                print("Epoch finished. Scheduler step to {}".format(self.current_epoch))
                self.lr_scheduler.step_epoch(self.current_epoch)
            else:
                print("Resuming from index {}".format(trained_idx))
                train_epoch(trained_idx, self)

    def finalize(self):
        self.writer.write("Stepping into final check")
        self.inference()

        self.writer.write(
            "Finished run in {}".format(self.total_timer.get_time_since_start())
        )
        self.checkpoint.finalize()

    def inference(self):
        if "val" in self.run_type and (
            (self.current_update % self.eval_interval != 0)
            or ("train" not in self.run_type)
        ):
            self.writer.write("Starting inference on val set")
            evaluate("val", self)

        if "inference" in self.run_type or "test" in self.run_type:
            self.writer.write("Starting inference on test set")
            evaluate("test", self)

        self.writer.write("The inference finished!")

    def _update_tensorboard(self, split):
        if self.running_config.tensorboard:
            stats_dict = self.meters[split].get_scalar_dict()
            self.tb_writer.add_scalars(stats_dict, self.current_update)

    def _print_log(self, split, stats={}):
        log_dict = collections.OrderedDict()
        log_dict.update(
            {f"progress on {split}": f"{self.current_update}/{self.max_update}"}
        )
        log_dict.update(self.meters[split].get_log_dict(split))
        log_dict["\n"] = "-" * 20
        log_dict.update(stats)

        self.writer.log_progress(log_dict)

    def _calculate_time_left(self):
        time_taken_for_log = time.time() * 1000 - self.timers["train"].start
        iterations_left = self.max_update - self.current_update
        num_logs_left = iterations_left / self.log_interval
        time_left = num_logs_left * time_taken_for_log

        eval_to_log_ratio = self.eval_iteration / self.log_interval
        num_eval_left = iterations_left / self.eval_interval
        time_left += num_eval_left * eval_to_log_ratio * time_taken_for_log

        return self.timers["train"].get_time_hhmmss(gap=time_left)

    def profile(self, text):
        if self.not_debug:
            return
        self.writer.write(text + ": " + self.profiler.get_time_since_start(), "debug")
        self.profiler.reset()

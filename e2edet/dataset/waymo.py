import os
import pickle
import copy

import numpy as np
import uuid
import torch

from e2edet.dataset import BaseDataset, register_task
from e2edet.dataset.helper import PointDetection, DataBaseSampler, collate3d


class UUIDGeneration:
    def __init__(self):
        self.mapping = {}

    def get_uuid(self, seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex
        return self.mapping[seed]


uuid_gen = UUIDGeneration()


@register_task("detection3d")
class WaymoDetection(BaseDataset):
    LABEL_TO_IDX = {
        "UNKNOWN": 0,
        "VEHICLE": 1,
        "PEDESTRIAN": 2,
        "SIGN": 3,
        "CYCLIST": 4,
    }
    IDX_TO_LABEL = ("UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST")

    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        if "name" in kwargs:
            dataset_name = kwargs["name"]
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
        else:
            dataset_name = "waymo"

        super().__init__(
            config,
            dataset_name,
            dataset_type,
            current_device=kwargs["current_device"],
            global_config=kwargs["global_config"],
        )

        self.root_path = self._get_absolute_path(imdb_file["root_path"])
        self.waymo_dataset = PointDetection(
            self._get_absolute_path(imdb_file["root_path"]),
            self._get_absolute_path(imdb_file["info_path"]),
            self.get_num_point_features(config["nsweeps"]),
            test_mode=(dataset_type != "train"),
            nsweeps=config["nsweeps"],
            load_interval=imdb_file["load_interval"],
        )

        self.db_sampler = None
        if imdb_file.get("db_sampler", None) is not None:
            db_sampler_config = imdb_file["db_sampler"]
            db_info_path = self._get_absolute_path(db_sampler_config["db_info_path"])
            with open(db_info_path, "rb") as file:
                db_info = pickle.load(file)
            groups = db_sampler_config["groups"]
            min_points = db_sampler_config["min_points"]
            difficulty = db_sampler_config["difficulty"]
            rate = db_sampler_config["rate"]

            self.db_sampler = DataBaseSampler(
                db_info, groups, min_points=min_points, difficulty=difficulty, rate=rate
            )

        classes = [self.LABEL_TO_IDX[name] for name in config["classes"]]
        self.classes = config["classes"]
        assert len(classes) > 0, "No classes found!"

        self.prepare = WaymoPreparation(classes, config["min_points"])
        self.pc_range = torch.tensor(config["pc_range"])

    def get_num_point_features(self, nsweeps):
        num_point_features = 5 if nsweeps == 1 else 6

        return num_point_features

    def get_answer_size(self):
        return len(self.LABEL_TO_IDX)

    def __len__(self):
        return len(self.waymo_dataset)

    @property
    def infos(self):
        return self.waymo_dataset.infos

    @property
    def nsweeps(self):
        return self.waymo_dataset.nsweeps

    def _load(self, idx):
        res, points, annos = self.waymo_dataset[idx]
        annos["labels"] = (
            np.array([self.LABEL_TO_IDX[name] for name in annos["names"]])
            .astype(np.int64)
            .reshape(-1)
        )

        sample = {
            "nsweeps": self.nsweeps,
            "calib": None,
        }
        target = {
            "metadata": res["metadata"],
            "boxes": annos["boxes"],
            "raw_boxes": annos["boxes"].copy(),
            "labels": annos["labels"],
            "raw_labels": annos["labels"].copy(),
            "num_points_in_gt": annos["num_points_in_gt"],
            "difficulty": annos["difficulty"],
        }

        points, target = self.prepare(points, target)

        if self.db_sampler is not None:
            sampled_dict = self.db_sampler.sample_all(
                self.root_path,
                annos["boxes"],
                annos["names"],
                res["metadata"]["num_point_features"],
            )

            if sampled_dict is not None:
                sampled_labels = np.array(
                    [self.LABEL_TO_IDX[name] for name in sampled_dict["gt_names"]]
                )
                sampled_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                target["boxes"] = np.concatenate(
                    [target["boxes"], sampled_boxes], axis=0
                )
                target["labels"] = np.concatenate(
                    [target["labels"], sampled_labels], axis=0
                )

                points = np.concatenate([sampled_points, points], axis=0)
        sample["points"] = points

        if self._dataset_type == "train":
            sample, target = self.train_processor(sample, target)
        else:
            sample, target = self.test_processor(sample, target)

        return sample, target

    def get_collate_fn(self):
        return collate3d

    @torch.no_grad()
    def prepare_for_evaluation(self, predictions, result_path, tracking=False):
        """
        Create a prediction objects file.
        """
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.protos import metrics_pb2

        objects = metrics_pb2.Objects()

        # reoganize infos
        new_infos = {}
        infos = self.infos
        for info in infos:
            token = info["token"]
            new_infos[token] = info
        infos = new_infos

        for token, prediction in predictions.items():
            info = infos[token]
            anno_path = os.path.join(self.root_path, info["anno_path"])
            with open(self._get_absolute_path(anno_path), "rb") as f:
                obj = pickle.load(f)

            box3d = prediction["pred_boxes3d"].numpy()
            scores = prediction["pred_scores"].numpy()
            labels = prediction["pred_labels"].numpy()

            if tracking:
                tracking_ids = prediction["tracking_ids"]

            for i in range(box3d.shape[0]):
                det = box3d[i]
                score = scores[i]
                label = labels[i]

                o = metrics_pb2.Object()
                o.context_name = obj["scene_name"]
                o.frame_timestamp_micros = int(obj["frame_name"].split("_")[-1])

                # Populating box and score
                box = label_pb2.Label.Box()
                box.center_x = det[0]
                box.center_y = det[1]
                box.center_z = det[2]
                box.length = det[3]
                box.width = det[4]
                box.height = det[5]
                box.heading = det[-1]

                o.object.box.CopyFrom(box)
                o.score = score
                # Use correct type
                o.object.type = label

                if tracking:
                    o.object.id = uuid_gen.get_uuid(int(tracking_ids[i]))

                objects.objects.append(o)

        # Write objects to a file
        if tracking:
            path = os.path.join(result_path, "tracking_pred.bin")
        else:
            path = os.path.join(result_path, "detection_pred.bin")

        print("results saved to {}".format(path))
        with open(path, "wb") as f:
            f.write(objects.SerializeToString())

    @torch.no_grad()
    def format_for_evalai(self, output, targets, local_eval=True, threshold=None):
        out_logits, out_bbox = output["pred_logits"], output["pred_boxes"]
        pc_range = self.pc_range.to(out_logits)
        pc_size = pc_range[3:] - pc_range[:3]

        out_bbox[..., :3] = out_bbox[..., :3] * pc_size + pc_range[:3]
        out_bbox[..., 3:6] = out_bbox[..., 3:6] * pc_size
        out_bbox[..., -1] = out_bbox[..., -1] * np.pi * 2 - np.pi
        out_prob = out_logits.sigmoid()
        out_prob = out_prob.view(out_logits.shape[0], -1)

        def _process_output(indices, bboxes):
            topk_boxes = indices.div(out_logits.shape[2], rounding_mode="floor")
            labels = indices % out_logits.shape[2]

            boxes = torch.gather(
                bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_bbox.shape[-1])
            )

            return labels, boxes

        if threshold is None:
            topk_values, topk_indices = torch.topk(out_prob, 125, dim=1, sorted=False)
            scores = topk_values
            labels, boxes = _process_output(topk_indices, out_bbox)

            results = [
                {
                    "scores": s.detach().cpu(),
                    "labels": l.detach().cpu(),
                    "boxes3d": b.detach().cpu(),
                }
                for s, l, b in zip(scores, labels, boxes)
            ]
        else:
            top_pred = out_prob > threshold
            indices = torch.arange(
                top_pred.shape[1], dtype=torch.int64, device=out_logits.device
            )
            results = []
            for i in range(out_logits.shape[0]):
                if not torch.any(top_pred[i]).item():
                    results.append(
                        {"scores": [], "labels": [], "boxes": [], "masks": []}
                    )
                    continue

                topk_indices = torch.masked_select(indices, top_pred[i])
                scores = torch.masked_select(out_prob[i], top_pred[i])

                labels, boxes = _process_output(topk_indices[None], out_bbox[i][None])
                results.append(
                    {"scores": scores[0], "labels": labels[0], "boxes": boxes[0]}
                )

        if local_eval:
            processed_results = {
                target["metadata"]["token"]: {
                    "pred_scores": output["scores"],
                    "pred_labels": output["labels"],
                    "pred_boxes3d": output["boxes3d"],
                    "metadata": target["metadata"],
                    "boxes3d": target["raw_boxes"].cpu(),
                    "labels": target["raw_labels"].cpu(),
                    "difficulty": target["difficulty"].cpu(),
                    "num_points_in_gt": target["num_points_in_gt"].cpu(),
                    "classes": copy.copy(self.classes),
                }
                for target, output in zip(targets, results)
            }
        else:
            processed_results = {
                target["metadata"]["token"]: {
                    "pred_scores": output["scores"],
                    "pred_labels": output["labels"],
                    "pred_boxes3d": output["boxes3d"],
                }
                for target, output in zip(targets, results)
            }

        return processed_results


class WaymoPreparation(object):
    def __init__(self, classes, min_points):
        super().__init__()
        self.classes = np.array(classes)
        self.min_points = min_points

    def __call__(self, points, target):
        if "boxes" in target:
            keep = (target["labels"][:, None] == self.classes).any(axis=1)
            keep = keep & (target["num_points_in_gt"] >= self.min_points)

            target["labels"] = target["labels"][keep]
            target["boxes"] = target["boxes"][keep]

        return points, target

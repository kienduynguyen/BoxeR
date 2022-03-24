import copy
import os
import math

import numpy as np

from e2edet.utils.distributed import (
    is_dist_avail_and_initialized,
    get_rank,
    get_world_size,
)
from e2edet.utils.det3d.general import box_collision_test
from e2edet.utils.det3d.box_ops import (
    center_to_corner_box2d,
    rotate_points_along_z,
)


class BatchSampler:
    def __init__(self, sampled_list, name=None, shuffle=True):
        num_replicas = 1
        rank = 0

        if is_dist_avail_and_initialized():
            num_replicas = get_world_size()
            rank = get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(sampled_list) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self._sampled_list = sampled_list
        self._indices = self._get_indices(shuffle)

        self._idx = 0
        self._name = name
        self._shuffle = shuffle

    def _get_indices(self, shuffle):
        indices = np.arange(len(self._sampled_list)).tolist()
        if shuffle:
            np.random.shuffle(indices)
        indices += indices[: (self.total_size - len(self._sampled_list))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return indices

    def _sample(self, num):
        if self._idx + num >= self.num_samples:
            ret = self._indices[self._idx :].copy()
            self._reset()
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


class DataBaseSampler:
    def __init__(self, db_infos, groups, min_points=0, difficulty=-1, rate=1.0):
        for k, v in db_infos.items():
            print(f"Loading {len(v)} {k} database infos")

        if min_points > 0 or difficulty > -1:
            print("Filtering database...")
            new_db_infos = {}
            for name, db_info in db_infos.items():
                new_db_infos[name] = [
                    info
                    for info in db_info
                    if info["num_points_in_gt"] > min_points
                    and info["difficulty"] >= difficulty
                ]

            db_infos = new_db_infos
            for k, v in db_infos.items():
                print(f"Loading {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []

        self._group_db_infos = self.db_infos  # just use db_infos

        for group_info in groups:
            group_names = list(group_info.keys())
            self._sample_classes += group_names
            self._sample_max_nums += list(group_info.values())

        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = BatchSampler(v, k)

    def sample_all(self, root_path, gt_boxes, gt_names, num_point_features):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self._sample_classes, self._sample_max_nums
        ):
            sampled_num = int(
                max_sample_num - np.sum([n == class_name for n in gt_names])
            )

            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(sampled_groups, sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class(
                    class_name, sampled_num, avoid_coll_boxes
                )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)

            s_points_list = []
            for info in sampled:
                info_path = os.path.join(root_path, info["path"])
                try:
                    s_points = np.fromfile(info_path, dtype=np.float32).reshape(
                        -1, num_point_features
                    )

                    if "rot_transform" in info:
                        rot = info["rot_transform"]
                        s_points[:, :3] = rotate_points_along_z(
                            s_points[np.newaxis, :, :], rot[np.newaxis, :]
                        )[0]
                    s_points[:, :3] += info["box3d_lidar"][:3]
                    s_points_list.append(s_points)
                    # print(pathlib.Path(info["path"]).stem)
                except Exception:
                    print(info_path)
                    continue

            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
            }
            ret["group_ids"] = np.arange(
                gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled)
            )
        else:
            ret = None

        return ret

    def sample(self, name, num):
        ret = self._sampler_dict[name].sample(num)
        return ret

    def sample_class(self, name, num, gt_boxes):
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_boxes.shape[0] :]
        sp_boxes_bv = center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, -1]
        )

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples

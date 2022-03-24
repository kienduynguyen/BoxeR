import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from e2edet.utils.det3d.general import read_from_file, read_pc_annotations


class PointDetection(Dataset):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(
        self,
        root_path,
        info_path,
        num_point_features,
        test_mode=False,
        nsweeps=1,
        load_interval=1,
    ):
        super(PointDetection, self).__init__()
        self.info_path = info_path
        self.root_path = root_path
        self.nsweeps = nsweeps
        self.load_interval = load_interval
        self.num_point_features = num_point_features

        self.test_mode = test_mode
        self._set_group_flag()
        self._load_infos()

    def _load_infos(self):
        with open(self.info_path, "rb") as f:
            infos_all = pickle.load(f)

        self.infos = infos_all[:: self.load_interval]
        # print("Using {} frames".format(len(self.infos)))

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def __getitem__(self, index):
        """This function is used for preprocess.
        you need to create a input dict in this function for network inference.
        format: {
            anchors
            voxels
            num_points
            coordinates
            if training:
                labels
                reg_targets
            [optional]anchors_mask, slow in SECOND v1.5, don't use this.
            [optional]metadata, in kitti, image index is saved in metadata
        }
        """
        return self.get_sensor_data(index)

    def __len__(self):
        if not hasattr(self, "infos"):
            self._load_infos()

        return len(self.infos)

    def get_sensor_data(self, idx):
        """Dataset must provide a unified function to get data.
        Args:
            idx: int or dict. this param must support int for training.
                if dict, should have this format (no example yet):
                {
                    sensor_name: {
                        sensor_meta
                    }
                }
                if int, will return all sensor data.
                (TODO: how to deal with unsynchronized data?)
        Returns:
            sensor_data: dict.
            if idx is int (return all), return a dict with all sensors:
            {
                sensor_name: sensor_data
                ...
                metadata: ... (for kitti, contains image_idx)
            }
            if sensor is lidar (all lidar point cloud must be concatenated to one array):
            e.g. If your dataset have two lidar sensor, you need to return a single dict:
            {
                "lidar": {
                    "points": ...
                    ...
                }
            }
            sensor_data: {
                points: [N, 3+]
                [optional]annotations: {
                    "boxes": [N, 7] locs, dims, yaw, in lidar coord system. must tested
                        in provided visualization tools such as second.utils.simplevis
                        or web tool.
                    "names": array of string.
                }
            }
            if sensor is camera (not used yet):
            sensor_data: {
                data: image string (array is too large)
                [optional]annotations: {
                    "boxes": [N, 4] 2d bbox
                    "names": array of string.
                }
            }
            metadata: {
                # dataset-specific information.
                # for kitti, must have image_idx for label file generation.
                image_idx: ...
            }
            [optional]calib # only used for kitti
        """
        info = self.infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps,
            },
            "metadata": {
                "image_prefix": self.root_path,
                "num_point_features": self.num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
        }

        if not os.path.isabs(self.infos[idx]["path"]):
            self.infos[idx]["path"] = os.path.join(
                self.root_path, self.infos[idx]["path"]
            )

        points = read_from_file(self.infos[idx], self.nsweeps)
        annos = read_pc_annotations(self.infos[idx])

        return res, points, annos

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import copy
import warnings

import omegaconf
import torch
from torch.utils.data.dataset import Dataset

from e2edet.dataset.processor import build_processor
from e2edet.utils.general import get_cache_dir, get_root


class BaseDataset(Dataset):
    def __init__(self, config, dataset_name, dataset_type="train", **kwargs):
        super().__init__()
        if config is None:
            config = {}
        self.config = config
        # self._iter_per_update = config.iter_per_update if dataset_type == "train" else 1
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self._device = kwargs["current_device"]
        self._global_config = kwargs["global_config"]
        self._iter_per_update = self._global_config.training.iter_per_update

    def _get_absolute_path(self, paths):
        if os.environ.get("E2E_DATASETS") is None:
            warnings.warn("E2E_DATASETS environment not found! Setting to '.data' ...")
            os.environ["E2E_DATASETS"] = get_cache_dir(".data")

        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                paths = os.path.join(os.environ.get("E2E_DATASETS"), paths)
            return paths
        else:
            raise TypeError(
                "Paths passed to dataset should either be " "string or list"
            )

    def init_processors(self):
        if not hasattr(self.config, "processors"):
            return

        for processor_key, processor_params in self.config.processors.items():
            if not processor_params:
                continue

            if "answer" in processor_key:
                processor_params = copy.deepcopy(processor_params)
                with omegaconf.open_dict(processor_params):
                    file_path = processor_params.params["class_file"]
                    processor_params.params["class_file"] = self._get_absolute_path(
                        file_path
                    )

            processor_object = build_processor(processor_params)
            setattr(self, processor_key, processor_object)

    def _prepare_batch(self, batch, non_blocking=False):
        sample, target = batch
        new_sample = {}
        new_target = []

        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                new_sample[k] = v.to(self._device, non_blocking=non_blocking)
            else:
                new_sample[k] = v

        for t in target:
            new_item = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    new_item[k] = v.to(self._device, non_blocking=non_blocking)
                else:
                    new_item[k] = v
            new_target.append(new_item)

        return (new_sample, new_target)

    def prepare_batch(self, batch, non_blocking=False):
        """
        Transfer cpu tensors in batch to gpu
        :batch: (sample, target)
            sample: dict of tensors
            target: list of dict
        """
        if self._iter_per_update > 1:
            return [
                self._prepare_batch(split, non_blocking=non_blocking) for split in batch
            ]
        else:
            return self._prepare_batch(batch, non_blocking=non_blocking)

    def _load(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        sample, target = self._load(index)
        sample["iter_per_update"] = self.iter_per_update

        return sample, target

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def iter_per_update(self):
        return self._iter_per_update

    @dataset_name.setter
    def dataset_name(self, name):
        self._dataset_name = name

    def get_collate_fn(self):
        raise NotImplementedError

    @torch.no_grad()
    def prepare_for_evaluation(self, predictions, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def format_for_evalai(self, output, *args, **kwargs):
        raise NotImplementedError

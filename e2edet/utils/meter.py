# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from collections import deque

import torch


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.averaged_value_deque = deque(maxlen=self.window_size)
        self.batch_sizes = deque(maxlen=self.window_size)
        self.total_samples = 0
        self.total = 0.0
        self.count = 0

    def update(self, value, batch_size=1):
        self.count += 1

        self.deque.append(value * batch_size)
        self.averaged_value_deque.append(value)
        self.batch_sizes.append(batch_size)

        self.total_samples += batch_size
        self.total += value * batch_size

    @property
    def median(self):
        d = torch.tensor(list(self.averaged_value_deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        s = torch.tensor(list(self.batch_sizes))
        return d.sum().item() / s.sum().item()

    @property
    def global_avg(self):
        return self.total / self.total_samples

    def get_latest(self):
        return self.averaged_value_deque[-1]


class Meter:
    def __init__(self, window_size=20, delimiter=", "):
        self.meters = {}
        self.delimiter = delimiter
        self.window_size = window_size

    def update(self, update_dict, batch_size=1):
        for k, v in update_dict.items():
            if isinstance(v, torch.Tensor):
                if v.dim() != 0:
                    v = v.mean()
                v = v.item()
            assert isinstance(v, (float, int))
            if k not in self.meters:
                self.meters[k] = SmoothedValue(self.window_size)
            self.meters[k].update(v, batch_size)

    def update_from_meter(self, meter):
        for key, value in meter.meters.items():
            assert isinstance(value, SmoothedValue)
            if key not in self.meters:
                self.meters[key] = SmoothedValue(self.window_size)
            self.meters[key] = value

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def get_scalar_dict(self):
        scalar_dict = {}
        for k, v in self.meters.items():
            scalar_dict[k] = v.get_latest()

        return scalar_dict

    def get_log_dict(self, split):
        log_dict = {}
        for k, v in self.meters.items():
            if "train" == split:
                log_dict[k] = "{:.4f}".format(v.median)
                log_dict["{}/avg".format(k)] = "{:.4f}".format(v.avg)
            else:
                log_dict[k] = "{:.4f}".format(v.global_avg)
        return log_dict

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if "train" in name:
                loss_str.append(
                    "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
                )
            else:
                # In case of val print global avg
                loss_str.append("{}: {:.4f}".format(name, meter.global_avg))

        return self.delimiter.join(loss_str)

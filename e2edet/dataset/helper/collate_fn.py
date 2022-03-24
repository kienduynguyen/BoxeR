import collections
import re

import torch
from torch._six import string_classes


np_str_obj_array_pattern = re.compile(r"[SaUO]")


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def _collate_sample2d(sample):
    assert sample[0]["image"].ndim == 3

    if len(sample) == 1:
        return {"image": sample[0]["image"][None], "mask": None}

    new_sample = {}
    total_shape = (sample[i]["image"].shape for i in range(len(sample)))
    shape = (len(sample), *(max(elem) for elem in zip(*total_shape)))
    new_sample["image"] = sample[0]["image"].new_zeros(shape)
    b, h, w = shape[0], shape[2], shape[3]
    new_sample["mask"] = sample[0]["image"].new_ones(b, h, w).bool()

    for i, elem in enumerate(sample):
        c, h, w = elem["image"].shape
        new_sample["image"][i, :c, :h, :w].copy_(elem["image"])
        new_sample["mask"][i, :h, :w] = False

    return new_sample


def collate2d(batch):
    batch = list(zip(*batch))

    iter_per_update = batch[0][0].get("iter_per_update", 1)
    if iter_per_update == 1:
        new_batch = [_collate_sample2d(batch[0]), batch[1]]
    elif iter_per_update > 1:
        sample = batch[0]
        target = batch[1]

        batch_size = len(sample)

        assert batch_size % iter_per_update == 0
        split_size = batch_size // iter_per_update

        new_batch = [
            [
                _collate_sample2d(sample[i * split_size : (i + 1) * split_size]),
                target[i * split_size : (i + 1) * split_size],
            ]
            for i in range(iter_per_update)
        ]
    else:
        raise ValueError("iter_per_update should be greater than or equal to 1")

    return new_batch


def _collate_sample3d(sample):
    new_sample = {}

    fields = sample[0].keys()
    num_grid = len(sample[0]["voxels"])
    for field in fields:
        if sample[0][field] is None:
            new_sample[field] = None
            continue

        if field in ("voxels", "num_points_per_voxel", "num_voxels"):
            for i in range(num_grid):
                if num_grid == 1:
                    new_sample[field] = torch.cat(
                        [elem[field][i] for elem in sample], dim=0
                    )
                else:
                    new_sample[field + f"_{i}"] = torch.cat(
                        [elem[field][i] for elem in sample], dim=0
                    )
        elif field == "coordinates":
            for i in range(num_grid):
                batch_idx = torch.cat(
                    [
                        torch.ones(elem[field][i].shape[0], dtype=elem[field][i].dtype)
                        * j
                        for j, elem in enumerate(sample)
                    ],
                    dim=0,
                ).unsqueeze(1)
                data = torch.cat([elem[field][i] for elem in sample], dim=0)

                if num_grid == 1:
                    new_sample[field] = torch.cat([batch_idx, data], dim=1)
                else:
                    new_sample[field + f"_{i}"] = torch.cat([batch_idx, data], dim=1)
        elif field in ["points", "calib", "iter_per_update"]:
            continue
        elif field == "grid_shape":
            for i in range(num_grid):
                if num_grid == 1:
                    new_sample[field] = torch.stack(
                        [elem[field][i] for elem in sample], dim=0
                    )
                else:
                    new_sample[field + f"_{i}"] = torch.stack(
                        [elem[field][i] for elem in sample], dim=0
                    )
        else:
            new_sample[field] = torch.stack([elem[field] for elem in sample], dim=0)
    new_sample["num_grid"] = num_grid
    new_sample["batch_size"] = len(sample)

    return new_sample


def collate3d(batch):
    batch = list(zip(*batch))

    iter_per_update = batch[0][0].get("iter_per_update", 1)
    if iter_per_update == 1:
        new_batch = [_collate_sample3d(batch[0]), batch[1]]
    elif iter_per_update > 1:
        sample = batch[0]
        target = batch[1]

        batch_size = len(sample)

        assert batch_size % iter_per_update == 0
        split_size = batch_size // iter_per_update

        new_batch = [
            (
                _collate_sample3d(sample[i * split_size : (i + 1) * split_size]),
                target[i * split_size : (i + 1) * split_size],
            )
            for i in range(iter_per_update)
        ]
    else:
        raise ValueError("iter_per_update should be greater than or equal to 1")

    return new_batch

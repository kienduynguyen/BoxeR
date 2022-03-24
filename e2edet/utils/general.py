import os
import math
import copy
import collections
import re

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch._six import string_classes

from e2edet.utils.distributed import get_world_size
from e2edet.utils.box_ops import box_cxcywh_to_xyxy


def normalize_period(x, offset, period):
    return (x + offset * period) / period


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_proposal_pos_embed(proposals, hidden_dim):
    assert hidden_dim % proposals.shape[-1] == 0
    num_pos_feats = int(hidden_dim / proposals.shape[-1])
    temperature = 10000
    scale = 2 * math.pi

    dim_t = torch.arange(num_pos_feats, dtype=proposals.dtype, device=proposals.device)
    dim_t = temperature ** (2 * (dim_t.div(2, rounding_mode="floor")) / num_pos_feats)
    proposals = proposals * scale
    proposals = proposals.unbind(-1)

    pos = []
    for proposal in proposals:
        proposal = proposal[..., None] / dim_t
        proposal = torch.stack(
            (proposal[..., 0::2].sin(), proposal[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos.append(proposal)
    pos = torch.cat(pos, dim=-1)

    return pos


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def filter_grads(parameters):
    return [param for param in parameters if param.requires_grad]


def get_root():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.abspath(os.path.join(root_folder, ".."))

    return root_folder


def get_cache_dir(cache_dir):
    # If cache_dir path exists do not join to mmf root
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(get_root(), cache_dir)
    return cache_dir


def get_batch_size(batch_size):
    world_size = get_world_size()

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    return batch_size // world_size


def get_absolute_path(paths):
    # String check should be first as Sequence would pass for string too
    if isinstance(paths, str):
        if not os.path.isabs(paths):
            root_dir = get_root()
            paths = os.path.join(root_dir, paths)
        return paths
    elif isinstance(paths, collections.abc.Iterable):
        return [get_absolute_path(path) for path in paths]
    else:
        raise TypeError("Paths passed to dataset should either be " "string or list")


def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024))
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def print_model_parameters(model, writer, return_only=False):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        writer.write(
            "Total Parameters: {}. Trained Parameters: {}".format(
                total_params, trained_params
            )
        )
    return total_params, trained_params


def get_optimizer_parameters(model):
    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    )
    has_custom = (
        hasattr(model.module, "get_optimizer_parameters")
        if is_parallel
        else hasattr(model, "get_optimizer_parameters")
    )

    if has_custom:
        parameters = (
            model.module.get_optimizer_parameters()
            if is_parallel
            else model.get_optimizer_parameters()
        )
    else:
        parameters = filter_grads(model.parameters())

    return parameters


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def extract_grid(x, x_mask, boxes, grid_size=15, align_corners=False, roi_align=False):
    """
    Params:
    :x: (B, C, H, W)
    :x_mask: (B, H, W)
    :boxes: (B, L, 4)
    Return:
    :grid: (B, L, grid_size, grid_size, C)
    """
    b, l = boxes.shape[:2]
    c = x.shape[1]
    if b == 0:
        return torch.zeros(
            0, l, grid_size, grid_size, c, device=x.device, dtype=x.dtype
        )

    grid_size = grid_size * 2 if roi_align else grid_size

    if align_corners:
        indices = torch.arange(0, grid_size, device=x.device, dtype=x.dtype)
        step = 1.0 / (grid_size - 1)
    else:
        indices = 0.5 + torch.arange(0, grid_size, device=x.device, dtype=x.dtype)
        step = 1.0 / grid_size
    i, j = torch.meshgrid(indices, indices, indexing="ij")
    grid_indices = torch.stack([j, i], dim=-1)  # 7 x 7 x 2

    boxes = box_cxcywh_to_xyxy(boxes)
    if x_mask is not None:
        not_x_mask = ~x_mask
        size_h = not_x_mask[:, :, 0].sum(dim=1, dtype=x.dtype)
        size_w = not_x_mask[:, 0, :].sum(dim=1, dtype=x.dtype)
        h, w = x.shape[-2:]

        ratio_h = size_h / h
        ratio_w = size_w / w
        ratio = torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=-1)

        boxes = boxes * ratio.unsqueeze(1)

    boxes1, boxes2 = boxes.unsqueeze(-2).unsqueeze(-2).split(2, dim=-1)

    grid = grid_indices * step * (boxes2 - boxes1) + boxes1
    grid = grid * 2 - 1
    grid = grid.view(b, l, grid_size * grid_size, 2)

    grid = F.grid_sample(x, grid, align_corners=False)

    if roi_align:
        grid = grid.view(b, -1, l, grid_size // 2, 2, grid_size // 2, 2)
        grid = grid.max(-1)[0].max(-2)[0]
    else:
        grid = grid.view(b, -1, l, grid_size, grid_size)
    grid = grid.permute(0, 2, 3, 4, 1)

    return grid


def paste_grid(seg_mask, boxes, x_size):
    # seg_mask: l x 7 x 7
    # boxes: l x 4
    assert seg_mask.dim() == 3
    assert boxes.shape[0] == seg_mask.shape[0]
    nq = boxes.shape[0]

    h, w = x_size
    x1, y1, x2, y2 = boxes.unsqueeze(-2).unsqueeze(-2).unbind(-1)

    img_x = torch.arange(w, device=boxes.device, dtype=boxes.dtype) + 0.5
    img_y = torch.arange(h, device=boxes.device, dtype=boxes.dtype) + 0.5
    img_y, img_x = torch.meshgrid(img_y, img_x, indexing="ij")

    # l x h x w
    img_y = (img_y - y1) / (y2 - y1) * 2 - 1
    img_x = (img_x - x1) / (x2 - x1) * 2 - 1
    img_grid = torch.stack([img_x, img_y], dim=-1)
    img_grid = img_grid.view(nq, h, w, 2)

    img = F.grid_sample(seg_mask[:, None], img_grid, align_corners=False)
    img = img.view(nq, h, w)

    return img


def flatten_with_shape(tensor_list, mask_list):
    """
    Params:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]

    Return:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)
    """
    assert isinstance(tensor_list, collections.abc.Sequence)
    assert len(tensor_list) > 0

    N = len(tensor_list)
    tensor_shape = torch.zeros(N, 2, dtype=torch.int64, device=tensor_list[0].device)
    tensor_flatten = []

    if mask_list is not None:
        mask_flatten = []

    for i, tensor in enumerate(tensor_list):
        new_tensor = tensor.flatten(2).permute(0, 2, 1)
        tensor_flatten.append(new_tensor)

        if mask_list is not None:
            mask = mask_list[i]
            new_mask = mask.flatten(1)
            mask_flatten.append(new_mask)
            assert tensor.shape[2] == mask.shape[1]
            assert tensor.shape[3] == mask.shape[2]
        tensor_shape[i, 0] = tensor.shape[2]
        tensor_shape[i, 1] = tensor.shape[3]

    mask_flatten = torch.cat(mask_flatten, dim=1) if mask_list is not None else None
    tensor_flatten = torch.cat(tensor_flatten, dim=1)

    return tensor_flatten, mask_flatten, tensor_shape


def view_with_shape(tensor_flatten, mask_flatten, tensor_shape):
    """
    Params:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)

    Return:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]
    """
    chunk_sizes = (tensor_shape[:, 0] * tensor_shape[:, 1]).tolist()
    N = tensor_shape.shape[0]

    if tensor_flatten is None and mask_flatten is None:
        raise ValueError("Both tensor and mask are None")
    B = tensor_flatten.shape[0] if tensor_flatten is not None else mask_flatten.shape[0]

    if tensor_flatten is not None:
        tensor_list = torch.split(tensor_flatten, chunk_sizes, dim=1)

    if mask_flatten is not None:
        mask_list = torch.split(mask_flatten, chunk_sizes, dim=1)

    tensor2d_list = [] if tensor_flatten is not None else None
    mask2d_list = [] if mask_flatten is not None else None
    for i in range(N):
        H, W = tensor_shape[i].tolist()
        if tensor_flatten is not None:
            tensor2d_list.append(
                tensor_list[i].view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            )
        if mask_flatten is not None:
            mask2d_list.append(mask_list[i].view(B, H, W))

    return tensor2d_list, mask2d_list


def split_with_shape(tensor_flatten, mask_flatten, tensor_shape):
    """
    Params:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)

    Return:
    :tensor_list: [(B, H1 * W1, C), ..., (B, HN * WN, C)]
    :mask_list: [(B, H1 * W1), ..., (B, HN * WN)]
    """
    chunk_sizes = (tensor_shape[:, 0] * tensor_shape[:, 1]).tolist()

    if tensor_flatten is None and mask_flatten is None:
        raise ValueError("Both tensor and mask are None")

    if tensor_flatten is not None:
        tensor_list = torch.split(tensor_flatten, chunk_sizes, dim=1)
    else:
        tensor_list = None

    if mask_flatten is not None:
        mask_list = torch.split(mask_flatten, chunk_sizes, dim=1)
    else:
        mask_list = None

    return tensor_list, mask_list


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def data_to_tensor(data):
    data_type = type(data)

    if isinstance(data, torch.Tensor):
        return data
    elif (
        data_type.__module__ == "numpy"
        and data_type.__name__ != "str_"
        and data_type.__name__ != "string_"
    ):
        if data_type.__name__ == "ndarray" or data_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(data.dtype.str) is not None:
                return data

            return torch.as_tensor(data)
        elif data.shape == ():
            return torch.as_tensor([data.item()])

    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float32)
    elif isinstance(data, int):
        return torch.tensor([data])
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: data_to_tensor(value) for key, value in data.items()}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return data_type(*(data_to_tensor(elem) for elem in data))
    elif isinstance(data, collections.abc.Sequence):
        return [data_to_tensor(elem) for elem in data]

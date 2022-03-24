# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from typing import List

import torch
from torchvision.ops.boxes import box_area


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def box_cxcywh_to_xyxy(x) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_intersect(boxes1, boxes2) -> torch.Tensor:
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2) -> List[torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = box_intersect(boxes1, boxes2)

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou, union


def generalized_box_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks,
        (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros(0, 4, device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(h, dtype=torch.float)
    x = torch.arange(w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def iou_with_ign(boxes1, boxes2) -> torch.Tensor:
    """
    Computes the amount of overlap of boxes2 has within boxes1, which is handy for dealing with
    ignore areas. Hence, assume that boxes2 are ignore regions and boxes1 are anchor boxes, then
    we may want to know how much overlap the anchors have inside the ignore regions boxes2.
    boxes1: (M, 4) [x1, y1, x2, y2]
    boxes2: (N, 4) [x1, y1, x2, y2]
    mode: if 'elementwise', M needs to be equal to N and we compute
        intersection of M pairs in boxes1 and boxes2 elementwise. Otherwise,
        we compute intersection of NxM pairs.
    """
    area1 = box_area(boxes1)
    intersect = box_intersect(boxes1, boxes2)
    iou_w_ign = intersect / area1

    return iou_w_ign

# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import math

import torch
from torch import nn
from torch.nn import functional as F

from e2edet.utils.general import get_proposal_pos_embed


class FixedPositionEmbedding(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super(FixedPositionEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None, ref_size=None):
        # x: B x C x H x W
        # mask: B x H x W
        eps = 1e-6
        if mask is not None:
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=x.dtype)
            x_embed = not_mask.cumsum(2, dtype=x.dtype)
        else:
            size_h, size_w = x.shape[-2:]
            y_embed = torch.arange(1, size_h + 1, dtype=x.dtype, device=x.device)
            x_embed = torch.arange(1, size_w + 1, dtype=x.dtype, device=x.device)
            y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
            x_embed = x_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)
            y_embed = y_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (
            2 * dim_t.div(2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)

        return pos


class FixedBoxEmbedding(nn.Module):
    def __init__(self, hidden_dim, temperature=10000, normalize=False):
        super(FixedBoxEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x, mask=None, ref_size=4):
        eps = 1e-6
        if mask is not None:
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=x.dtype)
            x_embed = not_mask.cumsum(2, dtype=x.dtype)

            size_h = not_mask[:, :, 0].sum(dim=-1, dtype=x.dtype)
            size_w = not_mask[:, 0, :].sum(dim=-1, dtype=x.dtype)
        else:
            size_h, size_w = x.shape[-2:]
            y_embed = torch.arange(1, size_h + 1, dtype=x.dtype, device=x.device)
            x_embed = torch.arange(1, size_w + 1, dtype=x.dtype, device=x.device)
            y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
            x_embed = x_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)
            y_embed = y_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

            size_h = torch.tensor([size_h] * x.shape[0], dtype=x.dtype, device=x.device)
            size_w = torch.tensor([size_w] * x.shape[0], dtype=x.dtype, device=x.device)

        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)

        h_embed = ref_size / size_h
        w_embed = ref_size / size_w

        h_embed = h_embed.unsqueeze(1).unsqueeze(2).expand_as(x_embed)
        w_embed = w_embed.unsqueeze(1).unsqueeze(2).expand_as(x_embed)

        center_embed = torch.stack([x_embed, y_embed], dim=-1)
        size_embed = torch.stack([w_embed, h_embed], dim=-1)
        center = get_proposal_pos_embed(center_embed, self.hidden_dim)
        size = get_proposal_pos_embed(size_embed, self.hidden_dim)
        box = center + size

        return box.permute(0, 3, 1, 2)


def build_position_encoding(position_embedding_type, hidden_dim):
    if position_embedding_type == "fixed":
        N_steps = hidden_dim // 2
        # TODO find a better way of exposing other arguments
        position_embedding = FixedPositionEmbedding(N_steps, normalize=True)
    elif position_embedding_type == "fixed_box":
        position_embedding = FixedBoxEmbedding(hidden_dim, normalize=True)
    else:
        raise ValueError(f"not supported {position_embedding_type}")

    return position_embedding

# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.nn as nn

from e2edet.model import BaseDetectionModel, register_model
from e2edet.module import build_resnet, build_transformer, Detector
from e2edet.utils.general import filter_grads
from e2edet.utils.modeling import get_parameters


@register_model("detr")
class DETR(BaseDetectionModel):
    def __init__(self, config, num_classes, **kwargs):
        super().__init__(config, num_classes, global_config=kwargs["global_config"])
        self.hidden_dim = config["hidden_dim"]
        aux_loss = config["aux_loss"]
        num_queries = config["num_queries"]

        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.detector = Detector(self.hidden_dim, num_classes, aux_loss)

    def get_optimizer_parameters(self):
        backbone_groups = []
        transformer_groups = []

        backbone_param_group = {"params": filter_grads(self.backbone.parameters())}
        backbone_groups.append(backbone_param_group)

        transformer_param_group = get_parameters(self, lr_except=["backbone"])
        transformer_groups.extend(transformer_param_group)

        return (backbone_groups, transformer_groups)

    def _build(self):
        self.backbone = build_resnet(self.config.backbone)
        self.transformer = build_transformer(self.config.transformer)
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels[-1], self.hidden_dim, kernel_size=1
        )

    def forward(self, sample, target):
        out, pos = self.backbone(sample["image"], sample["mask"])
        feature, mask = out[-1]

        hidden_state = self.transformer(
            self.input_proj(feature), mask, self.query_embed.weight, pos[-1]
        )
        out = self.detector(hidden_state)

        return out

import math

import torch.nn as nn

from e2edet.model import BaseDetectionModel, register_model
from e2edet.module import (
    build_backbone3d,
    build_transformer,
    Detector3d,
    MultiDetector3d,
)
from e2edet.utils.general import filter_grads
from e2edet.utils.modeling import get_parameters


@register_model("boxer3d")
class BoxeR3D(BaseDetectionModel):
    def __init__(self, config, num_classes, **kwargs):
        super().__init__(config, num_classes, global_config=kwargs["global_config"])
        self.hidden_dim = config["hidden_dim"]
        self.deform_lr_multi = config["deform_lr_multi"]
        self.num_level = config["transformer"]["params"]["nlevel"]
        aux_loss = config["aux_loss"]

        self.enc_detector = MultiDetector3d(self.hidden_dim, 1, 3, False)
        self.detector = Detector3d(self.hidden_dim, num_classes, aux_loss)

    def get_optimizer_parameters(self):
        backbone_groups = []
        transformer_groups = []

        backbone_param_group = {"params": filter_grads(self.backbone.parameters())}
        backbone_groups.append(backbone_param_group)

        transformer_param_group = get_parameters(
            self,
            lr_multi=self.deform_lr_multi,
            lr_module=["linear_box"],
            lr_except=["backbone"],
        )
        transformer_groups.extend(transformer_param_group)

        return (backbone_groups, transformer_groups)

    def _build(self):
        self.backbone = build_backbone3d(self.config.backbone)
        self.transformer = build_transformer(self.config.transformer)

        in_channels = self.backbone.num_channels
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels[i], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
                for i in range(self.backbone.return_layers)
            ]
        )
        self._reset_parameters()

        self.transformer.encoder.detector = [self.enc_detector]

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        nn.init.constant_(self.detector.class_embed.bias, bias_value)
        nn.init.constant_(self.detector.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.detector.bbox_embed.layers[-1].bias, 0)

        nn.init.constant_(self.enc_detector.class_embed.bias, bias_value)
        nn.init.constant_(self.enc_detector.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.enc_detector.bbox_embed.layers[-1].bias, 0)

        for module in self.input_proj.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, sample, target=None):
        voxels, coords, num_points_per_voxel, input_shape = (
            sample["voxels"],
            sample["coordinates"],
            sample["num_points_per_voxel"],
            sample["grid_shape"][0],
        )
        batch_size = sample["batch_size"]

        out, pos = self.backbone(
            voxels, coords, num_points_per_voxel, batch_size, input_shape
        )

        if self.num_level != len(out):
            raise RuntimeError(
                "num_level should be equal to the number of output features "
                "(num_level: {}, # output features: {})".format(
                    self.num_level, len(out)
                )
            )

        features = []
        pos_encodings = []

        for i, (src, _) in enumerate(out):
            features.append(self.input_proj[i](src))
            pos_encodings.append(pos[i])

        outputs = self.transformer(features, pos_encodings)
        hidden_state, ref_windows, src_embed, src_ref_windows = outputs

        out = self.detector(hidden_state, ref_windows)

        if not self.inferencing:
            enc_out = self.enc_detector(src_embed[None], src_ref_windows[None])
            out["enc_outputs"] = [
                {
                    "pred_logits": enc_out["pred_logits"],
                    "pred_boxes": enc_out["pred_boxes"],
                }
            ]

        return out

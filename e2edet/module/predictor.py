import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.utils.general import inverse_sigmoid


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SegmentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
        super().__init__()
        self.num_layers = num_layers
        in_dim = [hidden_dim] * (num_layers - 1)
        layers = [
            nn.Sequential(
                nn.ConvTranspose2d(input_dim, hidden_dim, 2, stride=2), nn.ReLU(),
            )
        ]

        layers.extend(
            [
                nn.Sequential(
                    nn.Conv2d(n, k, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                )
                for n, k in zip(in_dim, in_dim)
            ]
        )
        layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        n, b, l, s, _, c = x.shape
        x = x.view(-1, s, s, c).permute(0, 3, 1, 2).contiguous()
        x = self.layers(x).view(n, b, l, -1, 2 * s, 2 * s)

        return x


class Detector(nn.Module):
    def __init__(self, hidden_dim, num_classes, aux_loss, use_focal, mask_mode="none"):
        super(Detector, self).__init__()
        assert mask_mode in ("none", "mask_v1", "mask_v2")
        self.aux_loss = aux_loss

        if use_focal:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.mask_mode = mask_mode
        if mask_mode == "mask_v1":
            self.mask_embed = SegmentMLP(
                hidden_dim, hidden_dim, num_classes, 2, kernel_size=1
            )
        elif mask_mode == "mask_v2":
            self.mask_embed = SegmentMLP(hidden_dim, hidden_dim, 1, 2, kernel_size=1)

    def forward(self, x, ref_windows=None, roi=None, x_mask=None):
        """
        pred_logits: [batch_size x num_queries x (num_classes + 1)]
            the classification logits (including no-object) for all queries.
        pred_boxes: The normalized boxes coordinates for all queries, represented as
                    (center_x, center_y, width, height). These values are normalized in [0, 1],
                    relative to the size of each individual image (disregarding possible padding).
        """
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x)

        n, b, l = x.shape[:3]
        if self.mask_mode == "mask_v1":
            assert roi is not None, "roi should not be None!"

            outputs_mask = self.mask_embed(roi)
            top_labels = torch.max(outputs_class, dim=-1, keepdim=True)[1]
            mask_size = outputs_mask.shape[-2]
            top_labels = (
                top_labels.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, -1, mask_size, mask_size)
            )
            outputs_mask = torch.gather(outputs_mask, 3, top_labels).squeeze(3)
        elif self.mask_mode == "mask_v2":
            assert roi is not None, "roi should not be None!"

            outputs_mask = self.mask_embed(roi).squeeze(3)
        else:
            outputs_mask = None

        if ref_windows is not None:
            if ref_windows.shape[-1] == 4:
                inv_ref_windows = inverse_sigmoid(ref_windows)
                outputs_coord += inv_ref_windows
            else:
                raise ValueError("ref_windows should be 4 dim")

        if x_mask is not None:
            outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)
            outputs_coord.masked_fill(x_mask.unsqueeze(-1), -65504.0)
        outputs_coord = outputs_coord.sigmoid()

        if self.mask_mode != "none":
            out = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "pred_masks": outputs_mask[-1],
            }
        else:
            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_mask
            )

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_mode != "none":
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_masks": m}
                for a, b, m in zip(
                    outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1]
                )
            ]

        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class Detector3d(nn.Module):
    def __init__(self, hidden_dim, num_classes, aux_loss):
        super(Detector3d, self).__init__()
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 7, 3)

    def forward(self, x, ref_windows=None, x_mask=None):
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x) + inverse_sigmoid(ref_windows)
        outputs_coord = outputs_coord[..., [0, 1, 5, 2, 3, 6, 4]].sigmoid()

        if x_mask is not None:
            outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)
            outputs_coord.masked_fill(x_mask.unsqueeze(-1), 0)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class MultiDetector3d(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_references, aux_loss):
        super(MultiDetector3d, self).__init__()
        self.aux_loss = aux_loss
        self.num_references = num_references
        self.class_embed = nn.Linear(hidden_dim, num_references * num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, num_references * 7, 3)

    def forward(self, x, ref_windows=None, x_mask=None):
        """
        pred_logits: [batch_size x num_queries x (num_classes + 1)]
            the classification logits (including no-object) for all queries.
        pred_boxes: The normalized boxes coordinates for all queries, represented as
                    (center_x, center_y, width, height). These values are normalized in [0, 1],
                    relative to the size of each individual image (disregarding possible padding).
        """
        nl, b, l = x.shape[:3]
        ref_windows = ref_windows[..., : self.num_references, :]

        ref_windows_valid = (
            (ref_windows[..., :2] > 0.001) & (ref_windows[..., :2] < 0.999)
        ).all(-1)
        if x_mask is None:
            x_mask = ~ref_windows_valid
        else:
            x_mask = x_mask & (~ref_windows_valid)

        outputs_class = self.class_embed(x).view(nl, b, l, self.num_references, -1)
        outputs_coord = self.bbox_embed(x).view(nl, b, l, self.num_references, 7)

        if ref_windows is not None:
            if ref_windows.shape[-1] == 5:
                outputs_box, outputs_height = outputs_coord.split((5, 2), dim=-1)
                outputs_box = outputs_box + inverse_sigmoid(ref_windows)
                outputs_coord = torch.cat([outputs_box, outputs_height], dim=-1)
                outputs_coord = outputs_coord[..., [0, 1, 5, 2, 3, 6, 4]].contiguous()
            else:
                raise ValueError("ref_windows should be 4 dim")

        if x_mask is not None:
            outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)
            outputs_coord.masked_fill(x_mask.unsqueeze(-1), -65504.0)
        outputs_class = outputs_class.view(nl, b, l * self.num_references, -1)
        outputs_coord = outputs_coord.view(nl, b, l * self.num_references, -1).sigmoid()

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

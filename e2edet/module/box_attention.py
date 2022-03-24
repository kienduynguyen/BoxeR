import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.module.ops import BoxAttnFunction, InstanceAttnFunction


class InstanceAttention(nn.Module):
    def __init__(self, d_model, num_level, num_head, kernel_size):
        super(InstanceAttention, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"

        self.im2col_step = 64
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.kernel_size = kernel_size
        self.head_dim = d_model // num_head

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_head * 4, d_model)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_level * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_level * 4, d_model)
        )
        self.linear_attn_bias = nn.Parameter(torch.zeros(num_head * num_level * 4))

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._create_kernel_indices(self.kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / self.kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        offset_boxes = offset_boxes.view(b, l, self.num_head, self.num_level, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        boxes = ref_windows + offset_boxes / 8 * ref_windows[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        grid = center + self.kernel_indices * torch.relu(size)
        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = attn_weights.view(b, l1, self.num_head, self.num_level, 2, 2)
        attn_weights = attn_weights.repeat_interleave(self.kernel_size // 2, dim=-1)
        attn_weights = attn_weights.repeat_interleave(self.kernel_size // 2, dim=-2)

        spatial_attn_weights = attn_weights.view(b, l1, self.num_head, -1)
        spatial_attn_weights = F.softmax(spatial_attn_weights, dim=-1).view(
            b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
        )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)

        if not self.inferencing:
            level_attn_weights = attn_weights.view(
                b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
            )
            level_attn_weights = F.softmax(level_attn_weights, dim=3)

            output, mask_output = InstanceAttnFunction.apply(
                value,
                v_shape,
                v_start_index,
                sampled_grid,
                spatial_attn_weights,
                level_attn_weights,
                self.kernel_size,
                self.im2col_step,
            )
            attn_weights = (spatial_attn_weights, level_attn_weights)
            mask_output = self.out_proj(mask_output)
        else:
            output = BoxAttnFunction.apply(
                value,
                v_shape,
                v_start_index,
                sampled_grid,
                spatial_attn_weights,
                self.im2col_step,
            )
            attn_weights = (spatial_attn_weights,)
            mask_output = None
        output = self.out_proj(output)

        return output, mask_output, attn_weights


class BoxAttention(nn.Module):
    def __init__(self, d_model, num_level, num_head, kernel_size=2):
        super(BoxAttention, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"

        self.im2col_step = 64
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.head_dim = d_model // num_head
        self.kernel_size = kernel_size
        self.num_point = kernel_size ** 2

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_head * 4, d_model)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_level * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point, d_model)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point)
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / self.kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        offset_boxes = offset_boxes.view(b, l, self.num_head, self.num_level, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        boxes = ref_windows + offset_boxes / 8 * ref_windows[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        grid = center + self.kernel_indices * torch.relu(size)
        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(b, l1, self.num_head, -1), dim=-1)
        attn_weights = attn_weights.view(
            b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
        )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        output = BoxAttnFunction.apply(
            value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
        )
        output = self.out_proj(output)

        return output, attn_weights


class Box3dAttention(nn.Module):
    def __init__(self, d_model, num_level, num_head, with_rotation=True, kernel_size=2):
        super(Box3dAttention, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"

        num_variable = 5 if with_rotation else 4

        self.im2col_step = 64
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.head_dim = d_model // num_head
        self.with_rotation = with_rotation
        self.num_variable = num_variable
        self.kernel_size = kernel_size
        self.num_point = kernel_size ** 2

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_head * num_variable, d_model)
        )
        self.linear_box_bias = nn.Parameter(
            torch.zeros(num_head * num_level * num_variable)
        )

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point, d_model)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point)
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / 2
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        offset_boxes = offset_boxes.view(
            b, l, self.num_head, self.num_level, self.num_variable
        )

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
            ref_windows, ref_angles, _ = ref_windows.split((4, 1, 2), dim=-1)
        else:
            ref_windows = ref_windows.unsqueeze(3)
            ref_windows, ref_angles = ref_windows.split((4, 1), dim=-1)

        if self.with_rotation:
            offset_boxes, offset_angles = offset_boxes.split(4, dim=-1)
            angles = (ref_angles + offset_angles / 16) * 2 * math.pi
        else:
            angles = ref_angles.expand(b, l, self.num_head, self.num_level, 1)

        boxes = ref_windows + offset_boxes / 8 * ref_windows[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        rot_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
        rot_matrix = rot_matrix.view(b, l, self.num_head, self.num_level, 1, 2, 2)

        grid = self.kernel_indices * torch.relu(size)
        grid = center + (grid.unsqueeze(-2) * rot_matrix).sum(-1)

        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(b, l1, self.num_head, -1), dim=-1)
        attn_weights = attn_weights.view(
            b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
        )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        output = BoxAttnFunction.apply(
            value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
        )
        output = self.out_proj(output)

        return output, attn_weights

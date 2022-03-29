import math

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from e2edet.module.ops import InstanceAttnFunction
from e2edet.utils.general import view_with_shape


def PlainInstanceAttnFunction(
    value,
    value_spatial_shapes,
    sampling_locations,
    spatial_attention_weights,
    level_attention_weights,
    mask_size,
):
    """
    Params:
    :value: (B, L2, C)
    :value_spatial_shapes: (N, 2)
    :sampling_locations: (B, L1, nheads, nlevels, npoints, 2)
    :spatial_attention_weights: (B, L1, nheads, nlevels, mask_size, mask_size)
    :level_attention_weights: (B, L1, nheads, nlevels, mask_size, mask_size)
    Return:
    :o_samples: (B, L1, C)
    :v_samples: (B, L1, mask_size, mask_size, C)
    """
    b, l1, nheads, nlevels, npoints = sampling_locations.shape[:-1]
    assert npoints == (mask_size ** 2), "mask_points: {}, mask_size: {}".format(
        npoints, mask_size
    )
    assert mask_size % 2 == 0, "Only support even mask_size!"

    value, _ = view_with_shape(value, None, value_spatial_shapes)

    v_samples = []
    o_samples = []
    for level in range(nlevels):
        h, w = value[level].shape[2:]

        sampled_v = value[level].view(b * nheads, -1, h, w)
        grid = sampling_locations[..., level, :, :].transpose(1, 2)
        grid = grid.contiguous().view(b * nheads, l1, npoints, 2)

        sampled_v = F.grid_sample(sampled_v, grid, align_corners=False)
        sampled_v = sampled_v.view(b, nheads, -1, l1, mask_size, mask_size)
        # b x l1 x nheads x head_dim x mask_size x mask_size
        sampled_v = sampled_v.permute(0, 3, 1, 2, 4, 5)
        sampled_o = spatial_attention_weights[..., level : level + 1, :, :] * sampled_v
        sampled_o = sampled_o.sum(dim=-2).sum(dim=-1)
        o_samples.append(sampled_o)

        sampled_v = level_attention_weights[..., level : level + 1, :, :] * sampled_v
        v_samples.append(sampled_v)
    o_samples = torch.stack(o_samples, dim=1).sum(dim=1).contiguous()
    o_samples = o_samples.view(b, l1, -1)

    v_samples = torch.stack(v_samples, dim=1).sum(dim=1).contiguous()
    v_samples = v_samples.view(b, l1, -1, mask_size, mask_size).permute(0, 1, 3, 4, 2)

    return o_samples, v_samples


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 4
shapes = torch.tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
level_start_index = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])
MS = math.sqrt(P)
assert MS == float(int(MS))
MS = int(MS)

torch.manual_seed(3)


@torch.no_grad()
def check_forward(tensor_type="float"):
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, MS, MS).cuda() + 1e-5
    spatial_attention_weights = attention_weights / (
        attention_weights.sum(-1, keepdim=True)
        .sum(-2, keepdim=True)
        .sum(-3, keepdim=True)
    )
    level_attention_weights = attention_weights / attention_weights.sum(
        -3, keepdim=True
    )

    im2col_step = 2
    if tensor_type == "double":
        value = value.double()
        sampling_locations = sampling_locations.double()
        spatial_attention_weights = spatial_attention_weights.double()
        level_attention_weights = level_attention_weights.double()

    output_pytorch = PlainInstanceAttnFunction(
        value.view(N, S, -1),
        shapes,
        2 * sampling_locations - 1,
        spatial_attention_weights,
        level_attention_weights,
        MS,
    )
    output_cuda = InstanceAttnFunction.apply(
        value,
        shapes,
        level_start_index,
        sampling_locations,
        spatial_attention_weights,
        level_attention_weights,
        MS,
        im2col_step,
    )
    forward_check_0 = torch.allclose(
        output_cuda[0], output_pytorch[0], rtol=1e-2, atol=1e-3
    )

    max_abs_err_0 = (output_cuda[0] - output_pytorch[0]).abs().max()
    max_rel_err_0 = (
        (output_cuda[0] - output_pytorch[0]).abs() / output_pytorch[0].abs()
    ).max()

    print(
        f"{forward_check_0} check_forward_0 (tensor_type: {tensor_type}): max_abs_err {max_abs_err_0:.2e}, max_rel_err {max_rel_err_0:.2e}"
    )

    forward_check_1 = torch.allclose(
        output_cuda[1], output_pytorch[1], rtol=1e-2, atol=1e-3
    )

    max_abs_err_1 = (output_cuda[1] - output_pytorch[1]).abs().max()
    max_rel_err_1 = (
        (output_cuda[1] - output_pytorch[1]).abs() / output_pytorch[1].abs()
    ).max()

    print(
        f"{forward_check_1} check_forward_1 (tensor_type: {tensor_type}): max_abs_err {max_abs_err_1:.2e}, max_rel_err {max_rel_err_1:.2e}"
    )


def check_forward_and_backward(tensor_type="double"):
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, MS, MS).cuda() + 1e-5
    spatial_attention_weights = attention_weights / (
        attention_weights.sum(-1, keepdim=True)
        .sum(-2, keepdim=True)
        .sum(-3, keepdim=True)
    )
    level_attention_weights = attention_weights / attention_weights.sum(
        -3, keepdim=True
    )

    im2col_step = 2
    if tensor_type == "double":
        value = value.double()
        sampling_locations = sampling_locations.double()
        spatial_attention_weights = spatial_attention_weights.double()
        level_attention_weights = level_attention_weights.double()
    else:
        raise ValueError("only work with double tensor type")

    with torch.no_grad():
        pytorch_value = value.clone()
        pytorch_sampling_locations = sampling_locations.clone()
        pytorch_spatial_attention_weights = spatial_attention_weights.clone()
        pytorch_level_attention_weights = level_attention_weights.clone()

    value.requires_grad = True
    sampling_locations.requires_grad = True
    spatial_attention_weights.requires_grad = True
    level_attention_weights.requires_grad = True

    pytorch_value.requires_grad = True
    pytorch_sampling_locations.requires_grad = True
    pytorch_spatial_attention_weights.requires_grad = True
    pytorch_level_attention_weights.requires_grad = True

    output_pytorch = PlainInstanceAttnFunction(
        pytorch_value.view(N, S, -1),
        shapes,
        2 * pytorch_sampling_locations - 1,
        pytorch_spatial_attention_weights,
        pytorch_level_attention_weights,
        MS,
    )
    output_cuda = InstanceAttnFunction.apply(
        value,
        shapes,
        level_start_index,
        sampling_locations,
        spatial_attention_weights,
        level_attention_weights,
        MS,
        im2col_step,
    )

    (output_pytorch[0].sum() + output_pytorch[1].sum()).contiguous().backward()
    (output_cuda[0].sum() + output_cuda[1].sum()).contiguous().backward()

    forward_check_0 = torch.allclose(
        output_cuda[0], output_pytorch[0], rtol=1e-2, atol=1e-3
    )

    max_abs_err_0 = (output_cuda[0] - output_pytorch[0]).abs().max()
    max_rel_err_0 = (
        (output_cuda[0] - output_pytorch[0]).abs() / output_pytorch[0].abs()
    ).max()

    print(
        f"{forward_check_0} check_forward_0 (tensor_type: {tensor_type}): max_abs_err {max_abs_err_0:.2e}, max_rel_err {max_rel_err_0:.2e}"
    )

    forward_check_1 = torch.allclose(
        output_cuda[1], output_pytorch[1], rtol=1e-2, atol=1e-3
    )

    max_abs_err_1 = (output_cuda[1] - output_pytorch[1]).abs().max()
    max_rel_err_1 = (
        (output_cuda[1] - output_pytorch[1]).abs() / output_pytorch[1].abs()
    ).max()

    print(
        f"{forward_check_1} check_forward_1 (tensor_type: {tensor_type}): max_abs_err {max_abs_err_1:.2e}, max_rel_err {max_rel_err_1:.2e}"
    )

    backward_check1 = torch.allclose(
        value.grad, pytorch_value.grad, rtol=1e-2, atol=1e-3
    )
    backward_check2 = torch.allclose(
        sampling_locations.grad, pytorch_sampling_locations.grad, rtol=1e-2, atol=1e-3
    )
    backward_check3 = torch.allclose(
        spatial_attention_weights.grad,
        pytorch_spatial_attention_weights.grad,
        rtol=1e-2,
        atol=1e-3,
    )
    backward_check4 = torch.allclose(
        level_attention_weights.grad,
        pytorch_level_attention_weights.grad,
        rtol=1e-2,
        atol=1e-3,
    )
    backward_check = (
        backward_check1 and backward_check2 and backward_check3 and backward_check4
    )

    print(f"{backward_check} check_backward (tensor_type: {tensor_type})")


def check_gradient_numerical(
    channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
):
    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    spatial_attention_weights = attention_weights / (
        attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    )
    spatial_attention_weights = spatial_attention_weights.clone()
    level_attention_weights = attention_weights / (
        attention_weights.sum(-2, keepdim=True)
    )
    level_attention_weights = level_attention_weights.clone()

    im2col_step = 2
    func = InstanceAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    spatial_attention_weights.requires_grad = grad_attn_weight
    level_attention_weights.requires_grad = grad_attn_weight

    backward_check = gradcheck(
        func,
        (
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            spatial_attention_weights.double(),
            level_attention_weights.double(),
            MS,
            im2col_step,
        ),
    )

    print(f"{backward_check} check_gradient_numerical(D={channels})")


if __name__ == "__main__":
    try:
        for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
            check_gradient_numerical(channels, True, True, True)
    except Exception as e:
        print(e)

    check_forward("float")
    check_forward("double")
    check_forward_and_backward()

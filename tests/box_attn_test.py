import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from e2edet.module.ops import BoxAttnFunction
from e2edet.utils.general import view_with_shape


def PlainBoxAttnFunction(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Params:
    :value: (B, L2, C)
    :value_spatial_shapes: (N, 2)
    :sampling_locations: (B, L1, nheads, nlevels, npoints, 2)
    :attention_weights: (B, L1, nheads, nlevels, npoints)

    Return:
    :output: (B, L1, C)
    """
    b, l1, nheads, nlevels, npoints = attention_weights.shape
    value, _ = view_with_shape(value, None, value_spatial_shapes)

    v_samples = []
    for level in range(nlevels):
        h, w = value[level].shape[2:]

        sampled_v = value[level].view(b * nheads, -1, h, w)
        grid = sampling_locations[:, :, :, level].transpose(1, 2)
        grid = grid.contiguous().view(b * nheads, l1, npoints, 2)

        sampled_v = F.grid_sample(sampled_v, grid, align_corners=False)
        sampled_v = sampled_v.view(b, nheads, -1, l1, npoints).permute(0, 3, 1, 2, 4)
        sampled_v = (attention_weights[:, :, :, level].unsqueeze(-2) * sampled_v).sum(
            dim=-1
        )
        v_samples.append(sampled_v)
    v_samples = torch.stack(v_samples, dim=1).sum(dim=1).contiguous()
    v_samples = v_samples.view(b, l1, -1)

    return v_samples


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = torch.tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
level_start_index = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])


torch.manual_seed(3)


@torch.no_grad()
def check_forward(tensor_type="float"):
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

    im2col_step = 2
    if tensor_type == "double":
        value = value.double()
        sampling_locations = sampling_locations.double()
        attention_weights = attention_weights.double()

    output_pytorch = (
        PlainBoxAttnFunction(
            value.view(N, S, -1), shapes, 2 * sampling_locations - 1, attention_weights
        )
        .detach()
        .cpu()
    )
    output_cuda = (
        BoxAttnFunction.apply(
            value,
            shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        .detach()
        .cpu()
    )
    forward_check = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(
        f"{forward_check} check_forward (tensor_type: {tensor_type}): max_abs_err {max_abs_err:.2e}, max_rel_err {max_rel_err:.2e}"
    )


def check_forward_and_backward(tensor_type="double"):
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

    im2col_step = 2
    if tensor_type == "double":
        value = value.double()
        sampling_locations = sampling_locations.double()
        attention_weights = attention_weights.double()
    else:
        raise ValueError("only work with double tensor type")

    with torch.no_grad():
        pytorch_value = value.clone()
        pytorch_sampling_locations = sampling_locations.clone()
        pytorch_attention_weights = attention_weights.clone()

    value.requires_grad = True
    sampling_locations.requires_grad = True
    attention_weights.requires_grad = True
    pytorch_value.requires_grad = True
    pytorch_sampling_locations.requires_grad = True
    pytorch_attention_weights.requires_grad = True

    output_pytorch = PlainBoxAttnFunction(
        pytorch_value.view(N, S, -1),
        shapes,
        2 * pytorch_sampling_locations - 1,
        pytorch_attention_weights,
    )
    output_cuda = BoxAttnFunction.apply(
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ).contiguous()

    output_pytorch.sum().backward()
    output_cuda.sum().backward()

    forward_check = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(
        f"{forward_check} check_forward (tensor_type: {tensor_type}): max_abs_err {max_abs_err:.2e}, max_rel_err {max_rel_err:.2e}"
    )

    backward_check1 = torch.allclose(
        value.grad, pytorch_value.grad, rtol=1e-2, atol=1e-3
    )
    backward_check2 = torch.allclose(
        sampling_locations.grad, pytorch_sampling_locations.grad, rtol=1e-2, atol=1e-3
    )
    backward_check3 = torch.allclose(
        attention_weights.grad, pytorch_attention_weights.grad, rtol=1e-2, atol=1e-3
    )
    backward_check = backward_check1 and backward_check2 and backward_check3

    print(f"{backward_check} check_backward (tensor_type: {tensor_type})")


def check_gradient_numerical(
    channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
):
    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

    im2col_step = 2
    func = BoxAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    backward_check = gradcheck(
        func,
        (
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
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

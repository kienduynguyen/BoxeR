#pragma once
#include <torch/extension.h>

namespace e2edet {

#ifdef WITH_CUDA

std::vector<at::Tensor> instance_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &spatial_attn_weight,
    const at::Tensor &level_attn_weight,
    const int im2col_step
);

std::vector<at::Tensor> instance_attn_cuda_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &spatial_attn_weight,
    const at::Tensor &level_attn_weight,
    const at::Tensor &grad_output,
    const at::Tensor &grad_mask_output,
    const int im2col_step
);

#endif

std::vector<at::Tensor> instance_attn_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &spatial_attn_weight,
    const at::Tensor &level_attn_weight,
    const int im2col_step
)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return instance_attn_cuda_forward(
            value, 
            spatial_shapes,
            level_start_index,
            sampling_loc,
            spatial_attn_weight,
            level_attn_weight,
            im2col_step
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> instance_attn_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &spatial_attn_weight,
    const at::Tensor &level_attn_weight,
    const at::Tensor &grad_output,
    const at::Tensor &grad_mask_output,
    const int im2col_step
)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return instance_attn_cuda_backward(
            value, 
            spatial_shapes,
            level_start_index,
            sampling_loc,
            spatial_attn_weight,
            level_attn_weight,
            grad_output,
            grad_mask_output,
            im2col_step
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

}
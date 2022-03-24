import os
import re
import copy
from collections import OrderedDict

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 10):
    from torchvision._internally_replaced_utils import load_state_dict_from_url
else:
    from torchvision.models.utils import load_state_dict_from_url

from e2edet.utils.distributed import synchronize, is_master
from e2edet.module.position_encoding import build_position_encoding


model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale

        return x * scale + bias


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        stride_in_1x1=False,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = FrozenBatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride_1x1)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride_3x3, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        stride_in_1x1=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = FrozenBatchNorm2d
        self._norm_layer = norm_layer
        self.stride_in_1x1 = stride_in_1x1

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.stride_in_1x1,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    stride_in_1x1=self.stride_in_1x1,
                )
            )

        return nn.Sequential(*layers)

    def _replace_module(self, x):
        res = []
        for group in x:
            if group == "conv":
                res.append("bn")
            elif group in ["1", "2", "3"]:
                res.append(group)

        return "".join(res)

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = OrderedDict()
        replace_shortcut = {
            "shortcut.weight": "downsample.0.weight",
            "shortcut.norm": "downsample.1",
        }
        replace_layer = {
            "stem.": "",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
        }

        for key, value in state_dict.items():
            if key.startswith("fc"):
                continue
            new_key = re.sub(
                r"(stem\.|res2|res3|res4|res5)",
                lambda x: replace_layer[x.groups()[0]],
                key,
            )
            new_key = re.sub(
                r"(shortcut[.]weight|shortcut[.]norm)",
                lambda x: replace_shortcut[x.groups()[0]],
                new_key,
            )
            new_key = re.sub(
                r"(conv)([1-3])(.norm)",
                lambda x: self._replace_module(x.groups()),
                new_key,
            )
            new_state_dict[new_key] = value

        named_tuple = super().load_state_dict(new_state_dict, strict=strict)

        return named_tuple

    def _forward_impl(self, x, mask):
        return NotImplementedError

    def forward(self, x, mask):
        return self._forward_impl(x, mask)


class BackBone(ResNet):
    def __init__(
        self,
        block,
        layers,
        num_channels,
        position_encoding=None,
        freeze_backbone=None,
        return_interm_layers=None,
        **kwargs,
    ):
        hidden_dim = kwargs.pop("hidden_dim")
        self.ref_size = kwargs.pop("ref_size", 4)
        super(BackBone, self).__init__(block, layers, **kwargs)

        if freeze_backbone is not None:
            self._freeze_modules(freeze_backbone)

        if return_interm_layers is not None:
            num_channels = [
                num_channels[layer] for layer in sorted(return_interm_layers)
            ]
        else:
            return_interm_layers = set(["layer4"])
            num_channels = [num_channels["layer4"]]

        self.position_encoding = None
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

        self.return_layers = return_interm_layers
        self.num_channels = num_channels

    def _freeze_modules(self, modules):
        for module in modules:
            m = getattr(self, module)
            for parameter in m.parameters():
                parameter.requires_grad_(requires_grad=False)

    def _forward_impl(self, x, mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out = []
        pos = []
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            x = getattr(self, layer)(x)
            if layer in self.return_layers:
                if mask is not None:
                    x_mask = F.interpolate(
                        mask[None].float(), size=x.shape[-2:]
                    ).bool()[0]
                else:
                    x_mask = None

                out.append((x, x_mask))
                if self.position_encoding is not None:
                    pos.append(
                        self.position_encoding(x, x_mask, self.ref_size).type_as(x)
                    )
                else:
                    pos.append(None)

        return out, pos


def _load_state_dict(arch, pretrained, pretrained_path, model_data_dir):
    assert not (
        pretrained is True and pretrained_path is not None
    ), "Both pretrained and pretrained_path are activated ({}, {})".format(
        pretrained, pretrained_path
    )

    state_dict = None
    if pretrained:
        # Prevent nodes from caching state_dicts at the same time
        if is_master():
            load_state_dict_from_url(model_urls[arch])
        synchronize()
        state_dict = load_state_dict_from_url(model_urls[arch])
    elif pretrained_path:
        if not os.path.isabs(pretrained_path):
            pretrained_path = os.path.join(model_data_dir, pretrained_path)

        if not os.path.exists(pretrained_path):
            raise ValueError(
                "pretrained_path ({}) doesn't exist".format(pretrained_path)
            )
        state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))

    return state_dict


def _build_model(layers, num_channels, state_dict=None, **kwargs):
    model = BackBone(Bottleneck, layers, num_channels, **kwargs)
    if state_dict is not None:
        named_tuple = model.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained resnet:", named_tuple)

    return model


def build_resnet(config):
    arch = config["type"]
    params = copy.deepcopy(config["params"])

    with omegaconf.open_dict(params):
        pretrained = params.pop("pretrained", True)
        pretrained_path = params.pop("pretrained_path", None)
        model_data_dir = params.pop("model_data_dir", None)

        if arch == "resnet50":
            layers = [3, 4, 6, 3]
            num_channels = {
                "layer4": 2048,
                "layer3": 1024,
                "layer2": 512,
                "layer1": 256,
            }
        elif arch == "resnet50_dc5":
            layers = [3, 4, 6, 3]
            num_channels = {
                "layer4": 2048,
                "layer3": 1024,
                "layer2": 512,
                "layer1": 256,
            }
            params["replace_stride_with_dilation"] = [False, False, True]
            arch = "resnet50"
        elif arch == "resnet101":
            layers = [3, 4, 23, 3]
            num_channels = {
                "layer4": 2048,
                "layer3": 1024,
                "layer2": 512,
                "layer1": 256,
            }
        elif arch == "resnet101_dc5":
            layers = [3, 4, 23, 3]
            num_channels = {
                "layer4": 2048,
                "layer3": 1024,
                "layer2": 512,
                "layer1": 256,
            }
            params["replace_stride_with_dilation"] = [False, False, True]
            arch = "resnet101"
        else:
            if pretrained:
                raise RuntimeError(
                    "pretrained is only supported with arch of "
                    "resnet50|resnet101. Found {}".format(arch)
                )

    state_dict = _load_state_dict(arch, pretrained, pretrained_path, model_data_dir)
    model = _build_model(layers, num_channels, state_dict=state_dict, **params)

    return model

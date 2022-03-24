import torch
import torch.nn as nn

from e2edet.module.point_pillar import PointPillarsScatter, PillarFeatureNet
from e2edet.module.position_encoding import build_position_encoding


class ConvNet(nn.Module):
    def __init__(
        self, num_input_features, num_layers, ds_strides, ds_filters, norm_layer=None
    ):
        super(ConvNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.SyncBatchNorm
        self._norm_layer = norm_layer

        assert len(num_layers) == len(ds_strides) == len(ds_filters)

        in_filters = [num_input_features] + list(ds_filters)[:-1]
        ds_layers = []

        for i, num_layer in enumerate(num_layers):
            layer = self._make_layer(
                in_filters[i], ds_filters[i], num_layer, stride=ds_strides[i]
            )
            ds_layers.append(layer)
        self.ds_layers = nn.ModuleList(ds_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(
                m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm, nn.BatchNorm1d)
            ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_channels = ds_filters

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False, padding=1),
                norm_layer(32, planes)
                if norm_layer == nn.GroupNorm
                else norm_layer(planes, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
        )

        for i in range(1, num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(planes, planes, 3, bias=False, padding=1),
                    norm_layer(32, planes)
                    if norm_layer == nn.GroupNorm
                    else norm_layer(planes, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Params:
        :x: (B, C, H, W)

        Return:
        :out: [(B, C, H1, W1), ..., (B, C, Hn, Wn)]
        """
        # print("shape: ", x.shape)
        out = []
        for i in range(len(self.ds_layers)):
            x = self.ds_layers[i](x)
            out.append((x, None))

        return out


class Backbone3d(nn.Module):
    def __init__(
        self,
        hidden_dim,
        reader,
        extractor,
        neck,
        ref_size,
        position_encoding,
        return_layers,
    ):
        super(Backbone3d, self).__init__()
        self.reader = reader
        self.extractor = extractor
        self.neck = neck
        self.position_encoding = None
        self.ref_size = ref_size
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

        self.return_layers = return_layers
        self.num_channels = (
            [extractor.num_channels]
            if neck is None
            else list(neck.num_channels)[-return_layers:]
        )

    def forward(
        self, voxels, coordinates, num_points_per_voxel, batch_size, input_shape
    ):
        out = self.reader(voxels, num_points_per_voxel, coordinates)
        out = self.extractor(out, coordinates, batch_size, input_shape)

        if self.neck is not None:
            out = self.neck(out)

            pos = []
            new_out = []
            for i, (x, mask) in enumerate(out):
                if i >= (len(out) - self.return_layers):
                    pos.append(
                        self.position_encoding(x, mask, self.ref_size).type_as(x)
                    )
                    new_out.append((x, mask))
        else:
            assert self.return_layers == 1, "return_layers should be 1 without neck"
            pos = [self.position_encoding(out, None, self.ref_size).type_as(out)]
            new_out = [(out, None)]

        return new_out, pos


def build_backbone3d(config):
    arch = config["type"]
    params = config["params"]

    hidden_dim = params["hidden_dim"]
    position_encoding = params["position_encoding"]
    ref_size = params["ref_size"]

    if arch == "pointpillar":
        return_layers = params["return_layers"]
        reader = PillarFeatureNet(**params["reader"])
        extractor = PointPillarsScatter(**params["extractor"])
        if params["neck"] is not None:
            neck = ConvNet(**params["neck"])
        else:
            neck = None

        model = Backbone3d(
            hidden_dim,
            reader,
            extractor,
            neck,
            ref_size,
            position_encoding,
            return_layers,
        )
    else:
        raise ValueError("Unsupported arch: {}".format(arch))

    return model

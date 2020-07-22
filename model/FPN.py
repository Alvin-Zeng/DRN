# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        # add name of module into lists, inner: 1x1 conv, layer: 3x3 conv
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            # add module named as *_block into self-module
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # process the last lowest resolution feat and first feed it into 1 x 1 conv
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        # exclude the last one and process the feat from the second highest layer feat
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv1d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv1d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


# def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
#     def make_conv(
#         in_channels, out_channels, kernel_size, stride=1, dilation=1
#     ):
#         conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=dilation * (kernel_size - 1) // 2,
#             dilation=dilation,
#             bias=False if use_gn else True
#         )
#         # Caffe2 implementation uses XavierFill, which in fact
#         # corresponds to kaiming_uniform_ in PyTorch
#         nn.init.kaiming_uniform_(conv.weight, a=1)
#         if not use_gn:
#             nn.init.constant_(conv.bias, 0)
#         module = [conv,]
#         if use_gn:
#             module.append(group_norm(out_channels))
#         if use_relu:
#             module.append(nn.ReLU(inplace=True))
#         if len(module) > 1:
#             return nn.Sequential(*module)
#         return conv
#
#     return make_conv
#
#
# def group_norm(out_channels, affine=True, divisor=1):
#     out_channels = out_channels // divisor
#     # dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
#     dim_per_gp = -1
#     # num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
#     num_groups = 32
#     # eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
#     eps = 1e-5
#     return torch.nn.GroupNorm(
#         get_group_gn(out_channels, dim_per_gp, num_groups),
#         out_channels,
#         eps,
#         affine
#     )
#
#
# def get_group_gn(dim, dim_per_gp, num_groups):
#     """get number of groups used by GroupNorm, based on number of channels."""
#     assert dim_per_gp == -1 or num_groups == -1, \
#         "GroupNorm: can only specify G or C/G."
#
#     if dim_per_gp > 0:
#         assert dim % dim_per_gp == 0, \
#             "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
#         group_gn = dim // dim_per_gp
#     else:
#         assert dim % num_groups == 0, \
#             "dim: {}, num_groups: {}".format(dim, num_groups)
#         group_gn = num_groups
#
#     return group_gn


if __name__ == '__main__':
    conv1d_func = conv_with_kaiming_uniform(False, False)
    output_channels = 256
    conv1d = conv1d_func(768, output_channels, kernel_size=1, stride=1, dilation=1)
    top_blocks_conv = conv1d_func(output_channels, output_channels, kernel_size=3, stride=2, dilation=1)
    # [batch, channel, length]
    input_data = torch.rand((32, 768, 16))
    feat = conv1d(input_data)
    results = [feat]
    for i, scale in enumerate([8, 4]):
        conv1d = conv1d_func(output_channels, output_channels, kernel_size=1, stride=2, dilation=1)
        feat = conv1d(results[-1])
        results.append(feat)
    lastLevelP6P7 = LastLevelP6P7(output_channels, output_channels)
    fpn = FPN([256, 256, 256], 256, conv1d_func, top_blocks=lastLevelP6P7)
    pyramid_result = fpn(results)
    a = 0



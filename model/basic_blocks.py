import torch
from torch import nn


def conv_with_kaiming_uniform(use_bn=True, use_relu=True, use_dropout=False):
    def make_conv(
        in_channels, out_channels, kernel_size=3, stride=1, dilation=1
    ):
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_bn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        # nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_bn:
            module.append(nn.BatchNorm1d(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if use_dropout:
            module.append(nn.Dropout(p=0.5))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    # dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    dim_per_gp = -1
    # num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    num_groups = 32
    # eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
    eps = 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )

def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn
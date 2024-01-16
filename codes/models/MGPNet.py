# 1.重新写过网络代码，保持(b, c, h, w)的次序
import torch.nn as nn
import torch
import einops
import torch.nn.functional as F


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, channels, height, width = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n c (gh fh) (gw fw) -> n c (gh gw) (fh fw)",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n c (gh gw) (fh fw) -> n c (gh fh) (gw fw)",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


class CALayer(nn.Module):
    """
        压缩-激励模块
    """

    def __init__(self, features, reduction, use_bias):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features // reduction, kernel_size=(1, 1),
                               bias=use_bias)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=features // reduction, out_channels=features, kernel_size=(1, 1),
                               bias=use_bias)
        self.activation2 = nn.Sigmoid()

    def forward(self, x):
        # x.shape = b, c, h, w
        y = torch.mean(x, dim=[2, 3], keepdim=True)  # y.shape = b, c, 1, 1
        y = self.conv1(y)
        y = self.activation1(y)
        y = self.conv2(y)
        y = self.activation2(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, features, reduction=4, lrelu_slope=0.2, use_bias=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=features)  # 只对channels轴上的信息做归一化
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=(3, 3), bias=use_bias,
                               padding=1)
        self.activation = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=(3, 3), bias=use_bias,
                               padding=1)
        self.CALayer = CALayer(features=features, reduction=reduction, use_bias=use_bias)

    def forward(self, x):
        # x.shape = b, c, h, w
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.CALayer(x)
        return x + shortcut


class BlockGatingUnit(nn.Module):
    """gMLP模块中门电路 + spatial proj那一块的内容"""

    def __init__(self, num_channels=6, block_size=(2, 2), use_bias=True):
        super(BlockGatingUnit, self).__init__()
        n = block_size[0] * block_size[1]
        self.layernorm = nn.LayerNorm(normalized_shape=num_channels // 2)
        self.linear = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=1, bias=use_bias)
        self.num_channels = num_channels

    def forward(self, x):
        # x.shape = b, c, gh*gw, fh*fw
        u, v = torch.split(x, self.num_channels // 2, dim=1)
        v = v.permute(0, 3, 2, 1)
        v = self.layernorm(v)
        v = self.linear(v)
        v = v.permute(0, 3, 2, 1)
        return u * (v + 1.)


class GridGatingUnit(nn.Module):
    """gMLP模块中门电路 + spatial proj那一块的内容"""

    def __init__(self, num_channels=6, grid_size=(2, 2), use_bias=True):
        super().__init__()
        m = grid_size[0] * grid_size[1]
        self.layernorm = nn.LayerNorm(normalized_shape=num_channels // 2)
        self.linear = nn.Conv2d(in_channels=m, out_channels=m, kernel_size=1, bias=use_bias)
        self.num_channels = num_channels

    def forward(self, x):
        # x.shape = b, c, gh*gw, fh*fw
        u, v = torch.split(x, self.num_channels // 2, dim=1)
        v = v.permute(0, 2, 3, 1)
        v = self.layernorm(v)

        v = self.linear(v)
        v = v.permute(0, 3, 1, 2)
        return u * (v + 1.)


class BlockGmlpLayer(nn.Module):
    """Block gMLP提供局部信息"""

    def __init__(self, block_size, num_channels=3, factor=2, use_bias=True):
        super().__init__()
        self.block_size = block_size
        self.layernorm = nn.LayerNorm(normalized_shape=num_channels)
        self.y_ope = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels * factor, kernel_size=1, bias=use_bias),
            nn.GELU(),
            BlockGatingUnit(use_bias=use_bias, block_size=block_size, num_channels=num_channels * factor),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, bias=use_bias)
        )

    def forward(self, x):
        # x.shape = n, c, gh*fh, gw*fw
        b, c, h, w = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        x = block_images_einops(x, patch_size=(fh, fw))
        # x.shape = n, c, gh*gw, fh*fw
        y = x.permute(0, 2, 3, 1)
        y = self.layernorm(y)
        y = y.permute(0, 3, 1, 2)
        y = self.y_ope(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class GridGmlpLayer(nn.Module):
    """Grid gMLP提供全局信息"""

    def __init__(self, grid_size, num_channels=3, factor=2, use_bias=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=num_channels)
        self.y_ope = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels * factor, kernel_size=1, bias=use_bias),
            nn.GELU(),
            GridGatingUnit(use_bias=use_bias, grid_size=grid_size, num_channels=num_channels * factor),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, bias=use_bias)
        )
        self.grid_size = grid_size

    def forward(self, x):
        # x.shape = n, c, gh*fh, gw*fw
        b, c, h, w = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x, patch_size=(fh, fw))
        # x.shape = n, c, gh*gw, fh*fw
        y = x.permute(0, 2, 3, 1)
        y = self.layernorm(y)
        y = y.permute(0, 3, 1, 2)
        y = self.y_ope(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
    """multi-axis gated MLP block."""

    def __init__(self, block_size, grid_size, use_bias=True, num_channels=3, factor=2):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=num_channels)
        self.x_ope = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels * factor, kernel_size=1, bias=use_bias),
            nn.GELU()
        )
        self.GridGmlpLayer = GridGmlpLayer(grid_size=grid_size, num_channels=num_channels * factor // 2,
                                           factor=factor, use_bias=use_bias)
        self.BlockGmlpLayer = BlockGmlpLayer(block_size=block_size, num_channels=num_channels * factor // 2,
                                             factor=factor, use_bias=use_bias)
        self.linear = nn.Conv2d(in_channels=num_channels * factor, out_channels=num_channels,
                                kernel_size=1, bias=use_bias)

    def forward(self, x):
        # x.shape = b, c, h, w
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.x_ope(x)
        u, v = torch.split(x, x.shape[1] // 2, dim=1)
        # global branch
        u = self.GridGmlpLayer(u)
        # local branch
        v = self.BlockGmlpLayer(v)
        x = torch.cat([u, v], dim=1)
        x = self.linear(x)
        return x + shortcut


class DownSample(nn.Module):
    """使用步长为2，大小为2*2的卷积核来做下采样"""

    def __init__(self, features):
        super().__init__()
        self.convDown = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=2, stride=2)

    def forward(self, x):
        return self.convDown(x)


class UpSample(nn.Module):
    """使用大小为2*2，步长超参数设置为2的反卷积来做上采样"""

    def __init__(self, features):
        super().__init__()
        self.convUp = nn.ConvTranspose2d(in_channels=features, out_channels=features, kernel_size=(2, 2), stride=2)

    def forward(self, x):
        return self.convUp(x)


class ChangeChannelsConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=use_bias)

    def forward(self, x):
        return self.conv(x)


class GeneralBlock(nn.Module):
    """Encoder/Decoder Block的通式
        包含了一个大的residual block，其中用MAB和RCAB来学residual part.
    """

    def __init__(self, features, num_groups=2, block_size=(2, 2), grid_size=(2, 2),
                 use_bias=True, reduction=4, factor=2):
        super().__init__()
        groups_list = []
        for i in range(num_groups):
            groups_list.append(ResidualSplitHeadMultiAxisGmlpLayer(block_size=block_size, grid_size=grid_size,
                                                                   use_bias=use_bias, num_channels=features,
                                                                   factor=factor))
            groups_list.append(RCAB(features=features, reduction=reduction, use_bias=use_bias))
        self.groups_list = nn.ModuleList(groups_list)

    def forward(self, x):
        # x.shape = b, c, h, w
        shortcut = x
        for op in self.groups_list:
            x = op.forward(x)
        x = shortcut + x
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class MGPNet(nn.Module):
    def __init__(self, in_channels=3, size_lr=(8, 8), size_hr=(16, 16),
                 use_bias=True, factor=2, reduction=4, out_channels=3):
        super().__init__()
        # 各个尺度level规定的通道数
        features1 = 32
        features2 = 64
        features3 = 128
        # finest encoder
        EncoderBlock1 = [ChangeChannelsConv(in_channels=in_channels, out_channels=features1, use_bias=use_bias),
                         GeneralBlock(features=features1, block_size=size_hr, grid_size=size_hr,
                                      use_bias=use_bias, reduction=reduction, factor=factor)]
        self.EncoderBlock1 = nn.ModuleList(EncoderBlock1)
        self.ds1 = DownSample(features=features1)

        # middle encoder
        self.enConv2 = nn.Conv2d(in_channels=in_channels, out_channels=features1 // 2,
                                 kernel_size=3, padding=1, bias=use_bias)
        EncoderBlock2 = [ChangeChannelsConv(in_channels=features1 // 2 + features1,
                                            out_channels=features2, use_bias=use_bias),
                         GeneralBlock(features=features2, block_size=size_hr, grid_size=size_hr,
                                      use_bias=use_bias, reduction=reduction, factor=factor)]
        self.EncoderBlock2 = nn.ModuleList(EncoderBlock2)
        self.ds2 = DownSample(features=features2)

        # coarsest encoder
        self.enConv3 = nn.Conv2d(in_channels=in_channels, out_channels=features1 // 2,
                                 kernel_size=3, padding=1, bias=use_bias)
        EncoderBlock3 = [ChangeChannelsConv(in_channels=features1 // 2 + features2,
                                            out_channels=features3, use_bias=use_bias),
                         GeneralBlock(features=features3, block_size=size_lr, grid_size=size_lr,
                                      use_bias=use_bias, reduction=reduction, factor=factor)]
        self.EncoderBlock3 = nn.ModuleList(EncoderBlock3)

        # coarsest decoder
        DecoderBlock3 = [ChangeChannelsConv(in_channels=features3, out_channels=features3, use_bias=use_bias),
                         GeneralBlock(features=features3, block_size=size_lr, grid_size=size_lr,
                                      use_bias=use_bias, reduction=reduction, factor=factor)]
        self.DecoderBlock3 = nn.ModuleList(DecoderBlock3)
        self.deConv3 = nn.Conv2d(in_channels=features3, out_channels=out_channels, kernel_size=3,
                                 padding=1, bias=use_bias)
        self.us3 = UpSample(features=features3)

        # middle decoder
        DecoderBlock2 = [ChangeChannelsConv(in_channels=features1 // 2 + features3, out_channels=features2),
                         GeneralBlock(features=features2, block_size=size_hr, grid_size=size_hr,
                                      use_bias=use_bias, reduction=reduction, factor=factor)]
        self.DecoderBlock2 = nn.ModuleList(DecoderBlock2)
        self.deConv2 = nn.Conv2d(in_channels=features2, out_channels=out_channels, kernel_size=3,
                                 padding=1, bias=use_bias)
        self.us2 = UpSample(features=features2)

        # coarsest decoder
        DecoderBlock1 = [ChangeChannelsConv(in_channels=features1 // 2 + features2, out_channels=features1),
                         GeneralBlock(features=features1, block_size=size_hr, grid_size=size_hr,
                                      use_bias=use_bias, reduction=reduction, factor=factor)]
        self.DecoderBlock1 = nn.ModuleList(DecoderBlock1)
        self.deConv1 = nn.Conv2d(in_channels=features1, out_channels=out_channels, kernel_size=3,
                                 padding=1, bias=use_bias)

        self.AFF1 = AFF(in_channel=features1 + features2 + features3, out_channel=features1 // 2)
        self.AFF2 = AFF(in_channel=features1 + features2 + features3, out_channel=features1 // 2)

    def forward(self, x):
        # 有三个不同尺度的输入
        big = x
        mid = F.interpolate(big, scale_factor=0.5)
        small = F.interpolate(mid, scale_factor=0.5)
        # encoder1
        for op in self.EncoderBlock1:
            x = op.forward(x)
        skip1 = x
        skip1_AFF2 = F.interpolate(skip1, scale_factor=0.5)
        x = self.ds1(x)

        # encoder2
        x = torch.cat([x, self.enConv2(mid)], dim=1)
        for op in self.EncoderBlock2:
            x = op.forward(x)
        skip2 = x
        skip2_AFF1 = F.interpolate(skip2, scale_factor=2)
        x = self.ds2(x)

        # encoder3
        x = torch.cat([x, self.enConv3(small)], dim=1)
        for op in self.EncoderBlock3:
            x = op.forward(x)
        skip3 = x
        skip3_AFF2 = F.interpolate(skip3, scale_factor=2)
        skip3_AFF1 = F.interpolate(skip3_AFF2, scale_factor=2)

        # skip AFF1
        skipBig = self.AFF1.forward(skip1, skip2_AFF1, skip3_AFF1)

        skipMid = self.AFF2.forward(skip1_AFF2, skip2, skip3_AFF2)

        # decoder3
        for op in self.DecoderBlock3:
            x = op.forward(x)

        r3 = self.deConv3(x)
        outsmall = small + r3

        x = self.us3(x)

        # decoder2
        x = torch.cat([x, skipMid], dim=1)
        for op in self.DecoderBlock2:
            x = op.forward(x)

        r2 = self.deConv2(x)
        outmid = mid + r2
        x = self.us2(x)

        # decoder1
        x = torch.cat([x, skipBig], dim=1)
        for op in self.DecoderBlock1:
            x = op.forward(x)

        r1 = self.deConv1(x)
        outbig = big + r1

        return outbig, outmid, outsmall

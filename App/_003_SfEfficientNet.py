from typing import List, Callable
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor
import torch
import torch.nn as nn


# 定义一个函数，用于对输入的张量x进行通道排序，排序后的张量形状为[batch_size, num_channels, height, width]
# 参数：x：输入张量，groups：分组数
# 返回：排序后的张量
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    # 获取输入张量的形状
    batch_size, num_channels, height, width = x.size()
    # 计算每个组中包含的通道数
    channels_per_group = num_channels // groups

    # reshape
    # 将输入张量按照通道排序，排序后的张量形状为[batch_size, groups, channels_per_group, height, width]
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # 对输入张量按照通道进行转置，排序后的张量形状为[batch_size, channels_per_group, groups, height, width]
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    # 将输入张量按照通道进行折叠，排序后的张量形状为[batch_size, num_channels, height, width]
    # [batch_size, channels_per_group, groups, height, width] -> [batch_size, -1, height, width]
    x = x.view(batch_size, -1, height, width)

    return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        # 判断输入输出通道数是否为1或者2
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        # 判断输出通道数是否为偶数
        assert output_c % 2 == 0
        branch_features = output_c // 2

        # 判断输入输出通道数是否满足条件
        assert (self.stride != 1) or (input_c == branch_features << 1)

        # 如果输出步长为2，则构建分支1
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                # 深度卷积，输入输出通道数，卷积核大小，步长，填充
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                # 批量归一化
                nn.BatchNorm2d(input_c),
                # 卷积，输入输出通道数，卷积核大小，步长，填充，偏置
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                # 批量归一化
                nn.BatchNorm2d(branch_features),
                # 激活函数
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        # 构建分支2
        self.branch2 = nn.Sequential(
            # 卷积，输入输出通道数，卷积核大小，步长，填充，偏置
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            # 批量归一化
            nn.BatchNorm2d(branch_features),
            # 激活函数
            nn.ReLU(inplace=True),
            # 深度卷积，输入输出通道数，卷积核大小，步长，填充
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            # 批量归一化
            nn.BatchNorm2d(branch_features),
            # 卷积，输入输出通道数，卷积核大小，步长，填充，偏置
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            # 批量归一化
            nn.BatchNorm2d(branch_features),
            # 激活函数
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        '''
        创建一个深度可分离卷积
        Args:
            input_c (int): 输入通道数
            output_c (int): 输出通道数
            kernel_s (int): 卷积核大小
            stride (int): 步长
            padding (int): 填充
            bias (bool): 是否使用偏置
        Returns:
            nn.Conv2d: 创建的深度可分离卷积
        '''
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        '''
        前向传播
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: 输出张量
        '''
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


# ResNet Model
class BasicBlock(nn.Module):
    expansion = 1   # 标记卷积核个数是否变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        '''
        in_channel:   输入特征矩阵的深度
        out_channel:  输出特征矩阵的深度
        stride:       步长
        downsample:   下采样
        '''
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# Efficient_Net_V2 Model
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result

# class ResNet(nn.Module):
#
#     def __init__(self,
#                  block,
#                  blocks_num,
#                  include_top=True,
#                  groups=1,
#                  width_per_group=64):
#         super(ResNet, self).__init__()
#         self.include_top = include_top
#         self.in_channel = 64
#
#         self.groups = groups
#         self.width_per_group = width_per_group
#
#         self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=2)
#         self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion))
#
#         layers = []
#         layers.append(block(self.in_channel,
#                             channel,
#                             downsample=downsample,
#                             stride=stride,
#                             groups=self.groups,
#                             width_per_group=self.width_per_group))
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel,
#                                 channel,
#                                 groups=self.groups,
#                                 width_per_group=self.width_per_group))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#
#         return x


class New_Model_N(nn.Module):
    def __init__(self,
                 # model_cnf: list,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual,

                 ):
        super(New_Model_N, self).__init__()

        # Fused-MBConv模块
        model_cnf1 = [[2, 3, 1, 1, 32, 32, 0, 0],
                      [2, 3, 2, 4, 32, 64, 0, 0]]

        for cnf in model_cnf1:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf1[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf1])
        block_id = 0
        blocks1 = []
        for cnf in model_cnf1:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks1.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        self.blocks1 = nn.Sequential(*blocks1)
        input_channels = 64

        # 残差模块
        # self.resnet = ResNet(BasicBlock, [2, 2])
        # 定义静态注释，用于mypy
        # 检查stages_repeats是否为3个正整数
        # if len(stages_repeats) != 3:
        #     raise ValueError("expected stages_repeats as list of 3 positive ints")
        # 检查stages_out_channels是否为5个正整数
        # if len(stages_out_channels) != 5:
        #     raise ValueError("expected stages_out_channels as list of 5 positive ints")
        # 将stages_out_channels赋值给_stage_out_channels
        self._stage_out_channels = stages_out_channels
        self.stage2: nn.Sequential


        # 定义静态注释，用于mypy
        self.stage2: nn.Sequential
        # self.stage3: nn.Sequential
        # self.stage4: nn.Sequential

        # 定义stage2、stage3、stage4的名称
        stage_names = ["stage{}".format(i) for i in [2]]
        # 遍历stage_names，repeats、output_channels
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[0:]):
            # 定义seq，用于存放inverted_residual函数的输出
            seq = [inverted_residual(input_channels, output_channels, 2)]
            # 遍历range(repeats - 1)
            for i in range(repeats - 1):
                # 添加inverted_residual函数的输出到seq
                seq.append(inverted_residual(output_channels, output_channels, 1))
            # 将seq赋值给name对应的属性
            setattr(self, name, nn.Sequential(*seq))
            # 将输出通道数赋值给input_channels
            input_channels = output_channels

            # 定义stage5的输出通道数
            output_channels = self._stage_out_channels[-1]
            # 定义卷积层，输入通道数为input_channels，输出通道数为output_channels，卷积核大小为1，步长为1，填充为0，不使用偏置，激活函数为ReLU
            self.conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )


        # MBConv模块
        model_cnf2 = [[2, 5, 1, 6, 128, 160, 1, 0.25],
                      [2, 5, 2, 6, 160, 256, 1, 0.25]]
        total_blocks = sum([i[0] for i in model_cnf2])
        block_id = 0
        blocks2 = []
        for cnf in model_cnf2:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks2.append(op(kernel_size=cnf[1],
                                  input_c=cnf[4] if i == 0 else cnf[5],
                                  out_c=cnf[5],
                                  expand_ratio=cnf[3],
                                  stride=cnf[2] if i == 0 else 1,
                                  se_ratio=cnf[-1],
                                  drop_rate=drop_connect_rate * block_id / total_blocks,
                                  norm_layer=norm_layer))
                block_id += 1
        self.blocks2 = nn.Sequential(*blocks2)

        head_input_c = model_cnf2[-1][-3]
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks1(x)
        x = self.stage2(x)
        x = self.conv5(x)
        x = self.blocks2(x)
        x = self.head(x)

        return x


def SfEfficientNet(num_classes: int = 1000):
    model = New_Model_N(num_classes=num_classes,
                             stages_repeats=[4],
                             stages_out_channels=[128, 128])
    return model


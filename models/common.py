# This file contains modules common to various models
import math

import torch
import torch.nn as nn

from utils.activations import MemoryEfficientSwish
from utils.general import non_max_suppression

'''
feature map尺寸计算公式： out_size = (in_size + 2*Padding - kernel_size)/strides + 1
卷积计算时map尺寸向下取整
池化计算时map尺寸向上取整
'''


def autopad(k, p=None):  # kernel, padding
    '''
    自动填充
    返回padding值
        kernel_size 为 int类型时 ：padding = k // 2（整数除法进行一次）
                        否则    : padding = [x // 2 for x in k]
    '''
    # Pad to 'same'
    if p is None:  # k是否为int类型，是则返回True
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    '''
    标准卷积层Conv
    包括Conv2d + BN + HardWish激活函数
    (self, in_channels, out_channels, kernel_size, stride, padding, groups, activation_flag)
    p=None时，out_size = in_size/strides
    '''

    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):  # 前向计算（有BN）
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):  # 前向融合计算（无BN）
        return self.act(self.conv(x))


class DWConv(Conv):
    """
    深度分离卷积层 Depthwise convolution：
        是G（group）CONV的极端情况；
        分组数量等于输入通道数量，即每个通道作为一个小组分别进行卷积，结果联结作为输出，Cin = Cout = g，没有bias项。
        c1 : in_channels
        c2 : out_channels
        k : kernel_size
        s : stride
        act : 是否使用激活函数
        math.gcd() 返回的是最大公约数
    """
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    '''
    标准Bottleneck层
        input : input
        output : input + Conv3×3（Conv1×1(input)）
    (self, in_channels, out_channels, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''

    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        '''
        若 shortcut_flag为Ture 且 输入输出通道数相等，则返回跳接后的结构：
            x + Conv3×3（Conv1×1(x)）
        否则不进行跳接：
            Conv3×3（Conv1×1(x)）
        '''
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    '''
    标准bottleneckCSP层
    (self, in_channels, out_channels, Bottleneck层重复次数, shortcut_flag, group, expansion隐藏神经元的缩放因子)
    out_size = in_size
    '''

    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))  # CONV + BottleNeck + Conv2d  out_channels = c_
        y2 = self.cv2(x)  # Conv2d   out_channels = c_
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))  # concat(y1 + y2) + BN + LeakyReLU + Conv2d  out_channels = c2


# C3
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class SPP(nn.Module):
    '''
    空间金字塔池化SPP：
    (self, in_channels, out_channels, 池化尺寸strides[3])
    '''

    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 建立5×5 9×9 13×13的最大池化处理过程的list
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    '''
    Focus : 把宽度w和高度h的信息整合到c空间中
    (self, in_channels, out_channels, kernel_size, stride, padding, group, activation_flag)
    '''

    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        '''
        x(batch_size, channels, height, width) -> y(batch_size, 4*channels, height/2, weight/2)
        '''
        # ::代表[start:end:step], 以2为步长取值
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    '''
    (dimension)
    默认d=1按列拼接 ， d=0则按行拼接
    '''

    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Concat_BiFPN(nn.Module):
    # 不是直接concat了，而是进行pairwise add操作
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2, norm=True):
        super(Concat_BiFPN, self).__init__()
        # self.relu = nn.ReLU()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = DWConv(c1, c2, k=1, s=1)
        self.swish = MemoryEfficientSwish()
        # self.norm = norm
        # if self.norm:
        #     self.bn = nn.BatchNorm2d(num_features=c2, momentum=0.01, eps=1e-3)

    def forward(self, x):
        outs = self._forward(x)
        return outs

    def _forward(self, x):
        if len(x) == 2:
            # w = self.relu(self.w1)
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            # Connections for P6_0 and P7_0 to P6_1 respectively
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1]))
            # if self.norm:
            #     x = self.bn(x)
        elif len(x) == 3:
            # w = self.relu(self.w2)
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
            # if self.norm:
            #     x = self.bn(x)
        return x


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.3  # confidence threshold
    iou = 0.6  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, dimension=1):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class Flatten(nn.Module):
    '''
    在全局平均池化以后使用，去掉2个维度
    (batch_size, channels, size, size) -> (batch_size, channels*size*size)
    '''

    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    '''
    (self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1)
    (batch_size, channels, size, size) -> (batch_size, channels*1*1)
    '''

    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        # 给定输入数据和输出数据的大小，自适应算法能够自动帮助我们计算核的大小和每次移动的步长
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(batch_size,ch_in,1,1) 返回1×1的池化结果
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(batch_size,ch_out,1,1)
        self.flat = Flatten()

    def forward(self, x):
        #
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if x is list
        return self.flat(self.conv(z))  # flatten to x(batch_size, ch_out×1×1)


# SELayer
class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


# Coordinate Attention
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class Conv_CA(nn.Module):
    '''
    标准卷积层Conv + CA注意力机制
    包括Conv2d + BN + HardWish激活函数
    (self, in_channels, out_channels, kernel_size, stride, padding, groups, activation_flag)
    p=None时，out_size = in_size/strides
    '''

    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_CA, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = CoordAtt(c2, c2)

    def forward(self, x):  # 前向计算（有BN）
        return self.ca(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):  # 前向融合计算（无BN）
        return self.ca(self.act(self.conv(x)))


# CBAM模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.shared_MLP = nn.Sequential(  # MLP
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.sigmoid = nn.Sigmoid()
        #self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        #self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, c1, c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x  # 相乘是意味着给输入的每一个通道作权重分配
        out = self.spatial_attention(out) * out  # 相乘相当于给input作了通道注意力权重和空间注意力权重分配
        return out


# ASFF Adaptively Spatial Feature Fusion的自适应特征


# Transformer
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)


    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

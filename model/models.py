import torch
from torch import nn
import torch.nn.functional as F


# class Block(nn.Module):
#     '''expand + depthwise + pointwise'''
#     def __init__(self, in_planes, out_planes, expansion, stride):
#         super(Block, self).__init__()
#         self.stride = stride

#         planes = expansion * in_planes
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_planes)

#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_planes != out_planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = out + self.shortcut(x) if self.stride==1 else out
#         return out


# class MobileNetV2(nn.Module):
#     # (expansion, out_planes, num_blocks, stride)
#     cfg = [(1,  16, 1, 1),
#            (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
#            (6,  32, 3, 2),
#            (6,  64, 4, 2),
#            (6,  96, 3, 1),
#            (6, 160, 3, 2),
#            (6, 320, 1, 1)]

#     def __init__(self, num_classes=200):
#         super(MobileNetV2, self).__init__()
#         # NOTE: change conv1 stride 2 -> 1 for CIFAR10
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.layers = self._make_layers(in_planes=32)
#         self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(1280)
#         self.linear = nn.Linear(1280, num_classes)

#     def _make_layers(self, in_planes):
#         layers = []
#         for expansion, out_planes, num_blocks, stride in self.cfg:
#             strides = [stride] + [1]*(num_blocks-1)
#             for stride in strides:
#                 layers.append(Block(in_planes, out_planes, expansion, stride))
#                 in_planes = out_planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.relu(self.bn2(self.conv2(out)))
#         # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
#         out = F.avg_pool2d(out, 7)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out



# def dwise_conv(ch_in, stride=1):
#     return (
#         nn.Sequential(
#             #depthwise
#             nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
#             nn.BatchNorm2d(ch_in),
#             nn.ReLU6(inplace=True),
#         )
#     )

# def conv1x1(ch_in, ch_out):
#     return (
#         nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU6(inplace=True)
#         )
#     )

# def conv3x3(ch_in, ch_out, stride):
#     return (
#         nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU6(inplace=True)
#         )
#     )

# class InvertedBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, expand_ratio, stride):
#         super(InvertedBlock, self).__init__()

#         self.stride = stride
#         assert stride in [1,2]

#         hidden_dim = ch_in * expand_ratio

#         self.use_res_connect = self.stride==1 and ch_in==ch_out

#         layers = []
#         if expand_ratio != 1:
#             layers.append(conv1x1(ch_in, hidden_dim))
#         layers.extend([
#             #dw
#             dwise_conv(hidden_dim, stride=stride),
#             #pw
#             conv1x1(hidden_dim, ch_out)
#         ])

#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.layers(x)
#         else:
#             return self.layers(x)

# class MobileNetV2(nn.Module):
#     def __init__(self, ch_in=3, n_classes=200):
#         super(MobileNetV2, self).__init__()

#         self.configs=[
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1]
#         ]

#         self.stem_conv = conv3x3(ch_in, 32, stride=2)

#         layers = []
#         input_channel = 32
#         for t, c, n, s in self.configs:
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
#                 input_channel = c

#         self.layers = nn.Sequential(*layers)

#         self.last_conv = conv1x1(input_channel, 1280)

#         self.classifier = nn.Sequential(
#             nn.Dropout2d(0.2),
#             nn.Linear(1280, n_classes)
#         )
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         x = self.stem_conv(x)
#         x = self.layers(x)
#         x = self.last_conv(x)
#         x = self.avg_pool(x).view(-1, 1280)
#         x = self.classifier(x)
#         return x


# # if __name__=="__main__":
# #     # model check
# #     model = MobileNetV2(ch_in=3, n_classes=200)
# #     summary(model, (3, 224, 224), device='cpu')



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=200,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
import math
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


################################
###                          ### 
###           VGG16          ### 
###                          ###
################################


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, has_gate=False, struct=False):
        super(VGG, self).__init__()
        self.has_gate = has_gate
        self.struct = struct
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        if has_gate:
            self.pclassifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 10),
            )
            if self.struct:
                self.gate = nn.Linear(3 * 32 * 32, 1)
            else:
                self.gate = nn.Linear(512, 1)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        if self.has_gate:
            if self.struct:
                g = torch.sigmoid(self.gate(torch.flatten(input, 1)))
            else:
                g = torch.sigmoid(self.gate(x))
            y = self.classifier(x)
            z = self.pclassifier(x)
            return y * g + z * (1-g), g, z
        else:
            x = self.classifier(x)
            return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']



def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))



################################
###                          ### 
###       MobileNetV2        ### 
###                          ###
################################


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=200):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
    
################################
###                          ### 
###       MobileNetV3        ### 
###                          ###
################################
def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num
    
class H_sigmoid(nn.Module):
    '''
    hard sigmoid
    '''
    def __init__(self, inplace=True):
        super(H_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class H_swish(nn.Module):
    '''
    hard swish
    '''
    def __init__(self, inplace=True):
        super(H_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class SEModule(nn.Module):
    '''
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            H_sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y

class Bottleneck(nn.Module):
    '''
    The basic unit of MobileNetV3
    '''
    def __init__(self, in_channels_num, exp_size, out_channels_num, kernel_size, stride, use_SE, NL, BN_momentum):
        '''
        use_SE: True or False -- use SE Module or not
        NL: nonlinearity, 'RE' or 'HS'
        '''
        super(Bottleneck, self).__init__()

        assert stride in [1, 2]
        NL = NL.upper()
        assert NL in ['RE', 'HS']

        use_HS = NL == 'HS'
        
        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)

        if exp_size == in_channels_num:
            # Without expansion, the first depthwise convolution is omitted
            self.conv1 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels_num, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                    nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            )
        else:
            # With expansion
            self.conv1 = nn.Sequential(
                # Pointwise Convolution for expansion
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=exp_size, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                    nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            )

    def forward(self, x, expand=False):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        if self.use_residual:
            out = out + x
        if expand:
            return out, out1
        else:
            return out

class MobileNetV3(nn.Module):
    '''
    
    '''
    def __init__(self, mode='small', classes_num=1000, input_size=224, width_multiplier=1.0, dropout=0.2, BN_momentum=0.1, zero_gamma=False):
        '''
        configs: setting of the model
        mode: type of the model, 'large' or 'small'
        '''
        super(MobileNetV3, self).__init__()

        mode = mode.lower()
        assert mode in ['large', 'small']
        s = 2
        if input_size == 32 or input_size == 56:
            # using cifar-10, cifar-100 or Tiny-ImageNet
            s = 1

        # setting of the model
        if mode == 'large':
            # Configuration of a MobileNetV3-Large Model
            configs = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', s],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1]
            ]
        elif mode == 'small':
            # Configuration of a MobileNetV3-Small Model
            configs = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, True, 'RE', s],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        first_channels_num = 16

        # last_channels_num = 1280
        # according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # if small -- 1024, if large -- 1280
        last_channels_num = 1280 if mode == 'large' else 1024

        divisor = 8

        ########################################################################################################################
        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier, divisor) if width_multiplier > 1 else last_channels_num
        feature_extraction_layers = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channels_num, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channels_num, momentum=BN_momentum),
            H_swish()
        )
        feature_extraction_layers.append(first_layer)
        # Overlay of multiple bottleneck structures
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            exp_size = _ensure_divisible(exp_size * width_multiplier, divisor)
            feature_extraction_layers.append(Bottleneck(input_channels_num, exp_size, output_channels_num, kernel_size, stride, use_SE, NL, BN_momentum))
            input_channels_num = output_channels_num
        
        # the last stage
        last_stage_channels_num = _ensure_divisible(exp_size * width_multiplier, divisor)
        last_stage_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channels_num, out_channels=last_stage_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=last_stage_channels_num, momentum=BN_momentum),
                H_swish()
            )
        feature_extraction_layers.append(last_stage_layer1)

        
        self.featureList = nn.ModuleList(feature_extraction_layers)

        # SE Module
        # remove the last SE Module according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # feature_extraction_layers.append(SEModule(last_stage_channels_num) if mode == 'small' else nn.Sequential())

        last_stage = []
        last_stage.append(nn.AdaptiveAvgPool2d(1))
        last_stage.append(nn.Conv2d(in_channels=last_stage_channels_num, out_channels=last_channels_num, kernel_size=1, stride=1, padding=0, bias=False))
        last_stage.append(H_swish())

        self.last_stage_layers = nn.Sequential(*last_stage)
        

        ########################################################################################################################
        # Classification part
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channels_num, classes_num)
        )
        

        '''
        self.extras = nn.ModuleList([
            InvertedResidual(576, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
            InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.25)
        ])
        '''

        ########################################################################################################################
        # Initialize the weights
        self._initialize_weights(zero_gamma)

    def forward(self, x):
        for i in range(9):
            x = self.featureList[i](x)
        x = self.featureList[9](x)
        for i in range(10, len(self.featureList)):
            x = self.featureList[i](x)
        x = self.last_stage_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, zero_gamma):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for m in self.modules():
	            if hasattr(m, 'lastBN'):
	                nn.init.constant_(m.lastBN.weight, 0.0)

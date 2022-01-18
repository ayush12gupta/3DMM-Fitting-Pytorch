from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .arcface_torch.backbones import get_model
import math
from kornia.geometry import warp_affine

# __all__ = ['ResNet', 'resnet50']
def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1, bias = False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
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

    def __init__(self, block, layers=[3, 4, 6, 3], in_channels=3, out_channels=512, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, init_path, use_last_fc=False):
        last_dim = 2048
        super(Encoder, self).__init__()
        self.base = ResNet(Bottleneck, in_channels=in_channels, out_channels=last_dim, include_top=use_last_fc)  # resnet50
        state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
        self.base.load_state_dict(state_dict)
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True), # id layer
                conv1x1(last_dim, 64, bias=True), # exp layer
                conv1x1(last_dim, 80, bias=True), # tex layer
                conv1x1(last_dim, 3, bias=True),  # angle layer
                conv1x1(last_dim, 27, bias=True), # gamma layer
                conv1x1(last_dim, 2, bias=True),  # tx, ty
                conv1x1(last_dim, 1, bias=True)   # tz
            ])
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)
    
    def forward(self, x):
        x = self.base(x)
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        x = torch.flatten(torch.cat(output, dim=1), 1)
        return x

class RecogNetWrapper(nn.Module):
    def __init__(self, net_recog, pretrained_path=None, input_size=112):
        super(RecogNetWrapper, self).__init__()
        net = get_model(name=net_recog, fp16=False)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            net.load_state_dict(state_dict)
            print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
        
    def forward(self, image, M):
        image = self.preprocess(resize_n_crop(image, M, self.input_size))
        id_feature = F.normalize(self.net(image), dim=-1, p=2)
        return id_feature

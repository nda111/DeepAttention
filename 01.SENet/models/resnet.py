import os, sys

if __name__ == '__main__':
    sys.path.append(os.path.abspath(f'{__file__}/../../'))
    
from typing import Tuple

import torch
from torch import nn

from models.modules import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(
        self, block: type, num_blocks: Tuple[int, int, int, int], 
        num_classes: int=1000, use_se: bool=False, groups: int=1):
        
        super(ResNet, self).__init__()

        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.use_se = use_se
        self.groups = groups
        self.is_resnext = groups != 1

        self.tail = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        self.layer1 = self.__make_layer(64, 64, 256, downsample=False, num_blocks=num_blocks[0])
        self.layer2 = self.__make_layer(256, 128, 512, downsample=True, num_blocks=num_blocks[1])
        self.layer3 = self.__make_layer(512, 256, 1024, downsample=True, num_blocks=num_blocks[2])
        self.layer4 = self.__make_layer(1024, 512, 2048, downsample=True, num_blocks=num_blocks[3])

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, self.num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.tail(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

    def __make_layer(self, prev_out_channels: int, in_channels: int, out_channels: int, downsample: bool, num_blocks: int):
        first = [
            self.block(
                prev_out_channels, in_channels, downsample=downsample, 
                use_se=self.use_se, groups=self.groups)]

        middle = [
            self.block(
                in_channels, in_channels, downsample=False, 
                use_se=self.use_se, groups=self.groups)
            for _ in range(num_blocks - 2)]
            
        last = [
            self.block(
                in_channels, out_channels, downsample=False, 
                use_se=self.use_se, groups=self.groups)]

        layer = first + middle + last
        return nn.Sequential(*layer)


def resnet18(num_classes: int=1000, se: bool=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, use_se=se, groups=1)

def resnet34(num_classes: int=1000, se: bool=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, use_se=se, groups=1)

def resnet50(num_classes: int=1000, se: bool=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, use_se=se, groups=1)

def resnet101(num_classes: int=1000, se: bool=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, use_se=se, groups=1)

def resnet152(num_classes: int=1000, se: bool=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, use_se=se, groups=1)

def resnext50_32x4d(num_classes: int=1000, se: bool=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, use_se=se, groups=32)

def resnext101_32x8d(num_classes: int=1000, se: bool=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, use_se=se, groups=32)

def resnext101_64x4d(num_classes: int=1000, se: bool=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, use_se=se, groups=64)


def test():
    sample = torch.randn(4, 3, 224, 224)

    def case(net_type):
        net = net_type(se=False)
        out = net(sample)
        assert out.shape == (4, 1000)

        net = net_type(se=True)
        out = net(sample)
        assert out.shape == (4, 1000)

    case(resnet18)
    case(resnet34)
    case(resnet50)
    case(resnet101)
    case(resnet152)
    case(resnext50_32x4d)
    case(resnext101_32x8d)
    case(resnext101_64x4d)

    print('PASSED: /models/se_resnet.py')


if __name__ == '__main__':
    test()

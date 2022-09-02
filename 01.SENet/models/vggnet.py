import os, sys

if __name__ == '__main__':
    sys.path.append(os.path.abspath(f'{__file__}/../../'))

from typing import Tuple

import torch
from torch import nn

from models.modules import SEBlock


class VGGNet(nn.Module):
    def __init__(
        self, num_blocks: Tuple[int, int, int, int, int], use_bn: bool=True, 
        num_classes: int=1000, use_se: bool=False):
        
        super(VGGNet, self).__init__()

        self.num_blocks = num_blocks
        self.use_bn = use_bn
        self.num_classes = num_classes
        self.use_se = use_se

        self.layer1 = self.__make_layer(3, 64, num_blocks[0])
        self.layer2 = self.__make_layer(64, 128, num_blocks[1])
        self.layer3 = self.__make_layer(128, 256, num_blocks[2])
        self.layer4 = self.__make_layer(256, 512, num_blocks[3])
        self.layer5 = self.__make_layer(512, 512, num_blocks[4])

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, self.num_classes),
            nn.Softmax(dim=1),
        ) 
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.head(x)
        return x

    def __make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        def make_block(in_channels, out_channels):
            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(inplace=True),
            ]
            if self.use_bn:
                block.append(nn.BatchNorm2d(out_channels))
            if self.use_se:
                block.append(SEBlock(out_channels))
            return block

        layer = make_block(in_channels, out_channels)
        for _ in range(num_blocks - 1):
            layer.extend(make_block(out_channels, out_channels))

        layer.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        return nn.Sequential(*layer)


def vgg11(num_classes: int=1000, bn: bool=True, se: bool=False):
    return VGGNet([1, 1, 2, 2, 2], use_bn=bn, num_classes=num_classes, use_se=se)
    
def vgg13(num_classes: int=1000, bn: bool=True, se: bool=False):
    return VGGNet([2, 2, 2, 2, 2], use_bn=bn, num_classes=num_classes, use_se=se)
    
def vgg16(num_classes: int=1000, bn: bool=True, se: bool=False):
    return VGGNet([2, 2, 3, 3, 3], use_bn=bn, num_classes=num_classes, use_se=se)
    
def vgg19(num_classes: int=1000, bn: bool=True, se: bool=False):
    return VGGNet([2, 2, 3, 4, 4], use_bn=bn, num_classes=num_classes, use_se=se)


def test():
    sample = torch.randn(4, 3, 224, 224)

    def case(net_type):
        net = net_type(bn=False, se=False)
        out = net(sample)
        assert out.shape == (sample.size(0), net.num_classes)

        net = net_type(bn=False, se=True)
        out = net(sample)
        assert out.shape == (sample.size(0), net.num_classes)

        net = net_type(bn=True, se=False)
        out = net(sample)
        assert out.shape == (sample.size(0), net.num_classes)

        net = net_type(bn=True, se=True)
        out = net(sample)
        assert out.shape == (sample.size(0), net.num_classes)


    case(vgg11)
    case(vgg13)
    case(vgg16)
    case(vgg19)
    
    print('PASSED: /models/se_vgg16.py')
    

if __name__ == '__main__':
    test()

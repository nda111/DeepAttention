import os, sys
if __name__ == '__main__':
    sys.path.append(os.path.abspath(f'{__file__}/../../../'))

import torch
from torch import nn

from models.modules import SEBlock


def conv3x3(in_channels: int, out_channels: int, stride: int=1, groups: int=1):
    return nn.Conv2d(
        in_channels, out_channels,
         kernel_size=(3, 3),  
         padding=(1, 1),
         stride=stride, groups=groups, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int=1):
    return nn.Conv2d(
        in_channels, out_channels, 
        kernel_size=(1, 1), stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, out_channels: int, downsample: bool=False, 
        use_se: bool=False, groups: int=1):

        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.use_se = use_se

        stride = 2 if downsample else 1

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if self.use_se:
            self.se_block = SEBlock(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = torch.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = torch.relu(y)
        if self.use_se:
            y = self.se_block(y)

        x = self.conv3(x)
        x = self.bn3(x)
        return y + x
    

class Bottleneck(nn.Module):
    def __init__(
        self, 
        in_channels: int, out_channels: int, downsample: bool=False, 
        use_se: bool=False, groups: int=1):

        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.use_se = use_se

        stride = 2 if downsample else 1

        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn4 = nn.BatchNorm2d(out_channels)

        if self.use_se:
            self.se_block = SEBlock(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = torch.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = torch.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)
        y = torch.relu(y)
        if self.use_se:
            y = self.se_block(y)

        x = self.conv4(x)
        x = self.bn4(x)

        return y + x
    

def test():
    sample = torch.randn(4, 3, 112, 112)

    def case(block, use_se, is_resnext):
        net = nn.Sequential(
            block(3, 64, downsample=True, use_se=use_se, is_resnext=False),
            block(64, 64, downsample=False, use_se=use_se, is_resnext=is_resnext),
            block(64, 64, downsample=False, use_se=use_se, is_resnext=is_resnext),
        )
        out = net(sample)
        assert out.shape == (4, 64, 56, 56)

    case(BasicBlock, use_se=False, is_resnext=False)
    case(BasicBlock, use_se=False, is_resnext=True)
    case(BasicBlock, use_se=True, is_resnext=False)
    case(BasicBlock, use_se=True, is_resnext=True)
    case(Bottleneck, use_se=False, is_resnext=False)
    case(Bottleneck, use_se=False, is_resnext=True)
    case(Bottleneck, use_se=True, is_resnext=False)
    case(Bottleneck, use_se=True, is_resnext=True)

    print('PASSED: /models/modules/resnet.py')


if __name__ == '__main__':
    test()

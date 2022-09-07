import os, sys


if __name__ == '__main__':
    sys.path.append(os.path.abspath(f'{__file__}/../../'))

import torch
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from models.modules import SEBasicBlock, SEBottleneck

def resnet18(num_classes: int=1000, se: bool=False):
    block = SEBasicBlock if se else BasicBlock
    return ResNet(block, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes: int=1000, se: bool=False):
    block = SEBasicBlock if se else BasicBlock
    return ResNet(block, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes: int=1000, se: bool=False):
    block = SEBottleneck if se else Bottleneck
    return ResNet(block, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes: int=1000, se: bool=False):
    block = SEBottleneck if se else Bottleneck
    return ResNet(block, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes: int=1000, se: bool=False):
    block = SEBottleneck if se else Bottleneck
    return ResNet(block, [3, 8, 36, 3], num_classes=num_classes)

def resnext50_32x4d(num_classes: int=1000, se: bool=False):
    block = SEBottleneck if se else Bottleneck
    return ResNet(block, [3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4)

def resnext101_32x8d(num_classes: int=1000, se: bool=False):
    block = SEBottleneck if se else Bottleneck
    return ResNet(block, [3, 4, 23, 3], num_classes=num_classes, groups=32, width_per_group=8)

def resnext101_64x4d(num_classes: int=1000, se: bool=False):
    block = SEBottleneck if se else Bottleneck
    return ResNet(block, [3, 8, 36, 3], num_classes=num_classes, groups=64, width_per_group=4)

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

    print('PASSED: /models/resnet.py')


if __name__ == '__main__':
    test()

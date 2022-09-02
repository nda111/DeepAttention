import models as m

vgg11 = 'vgg11'
vgg13 = 'vgg13'
vgg16 = 'vgg16'
vgg19 = 'vgg19'

vgg11_bn = 'vgg11_bn'
vgg13_bn = 'vgg13_bn'
vgg16_bn = 'vgg16_bn'
vgg19_bn = 'vgg19_bn'

se_vgg11 = 'se_vgg11'
se_vgg13 = 'se_vgg13'
se_vgg16 = 'se_vgg16'
se_vgg19 = 'se_vgg19'

se_vgg11_bn = 'se_vgg11_bn'
se_vgg13_bn = 'se_vgg13_bn'
se_vgg16_bn = 'se_vgg16_bn'
se_vgg19_bn = 'se_vgg19_bn'

resnet18 = 'resnet18'
resnet34 = 'resnet34'
resnet50 = 'resnet50'
resnet101 = 'resnet101'
resnet152 = 'resnet152'
resnext50_32x4d = 'resnext50_32x4d'
resnext101_32x8d = 'resnext101_32x8d'
resnext101_64x4d = 'resnext101_64x4d'

se_resnet18 = 'se_resnet18'
se_resnet34 = 'se_resnet34'
se_resnet50 = 'se_resnet50'
se_resnet101 = 'se_resnet101'
se_resnet152 = 'se_resnet152'
se_resnext50_32x4d = 'se_resnext50_32x4d'
se_resnext101_32x8d = 'se_resnext101_32x8d'
se_resnext101_64x4d = 'se_resnext101_64x4d'

all_names = [
    vgg11, vgg13, vgg16, vgg19,
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
    se_vgg11, se_vgg13, se_vgg16, se_vgg19,
    se_vgg11_bn, se_vgg13_bn, se_vgg16_bn, se_vgg19_bn,
    resnet18, resnet34, resnet50, resnet101, resnet152,
    se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152,
    resnext50_32x4d, resnext101_32x8d, resnext101_64x4d,
    se_resnext50_32x4d, se_resnext101_32x8d, se_resnext101_64x4d,
]


def get_model(name: str, num_classes: int=1000):
    if name not in all_names:
        raise NotImplementedError('Unsupported model name.')

    name = name.lower().split('_')
    if name[0] == 'se':
        se = True
        name = name[1:]
    else:
        se = False
    
    if name[-1] == 'bn':
        bn = True
    else:
        bn = False

    name = name[0]
    if name == vgg11:
        return m.vgg11(num_classes, bn=bn, se=se)
    elif name == vgg13:
        return m.vgg13(num_classes, bn=bn, se=se)
    elif name == vgg16:
        return m.vgg16(num_classes, bn=bn, se=se)
    elif name == vgg19:
        return m.vgg19(num_classes, bn=bn, se=se)
    elif name == resnet18:
        return m.resnet18(num_classes, se=se)
    elif name == resnet34:
        return m.resnet34(num_classes, se=se)
    elif name == resnet50:
        return m.resnet50(num_classes, se=se)
    elif name == resnet101:
        return m.resnet101(num_classes, se=se)
    elif name == resnet152:
        return m.resnet152(num_classes, se=se)
    elif name == resnext50_32x4d:
        return m.resnext50_32x4d(num_classes, se=se)
    elif name == resnext101_32x8d:
        return m.resnext101_32x8d(num_classes, se=se)
    elif name == resnext101_64x4d:
        return m.resnext101_64x4d(num_classes, se=se)
    else:
        raise NotImplementedError('Unsupported model name.')

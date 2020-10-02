'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import sys
import models._modules as my_nn
from models import load_pre_state_dict

__all__ = [
    'cifar10_vggtiny', 'cifar10_vggtiny_bwn', 'cifar10_vggtiny_f_bwn',
    'cifar10_vggtiny_llsqs_bwns', 'cifar10_vggtiny_bn', 'cifar10_vggvtiny',
    'cifar10_vggvtiny_llsqs_bwns', 'cifar10_vggvvtiny', 'cifar10_vggvvtiny_llsqs_bwns'
]

model_urls = {
    'vgg7_bn': 'https://fake/models/cifar10-vggtiny-bn-93.29-zxd.pth',
    'vgg7_bn_ps': 'https://fake/models/cifar10-vggtiny-bn-ps-92.66-zxd.pth',
    'vgg7_ps': 'https://fake/models/cifar10-vggtiny-ps-92.66-zxd.pth',
    'vgg7': 'https://fake/models/cifar10-vggtiny-93.29-zxd.pth',
    'vggvtiny': 'https://fake/models/cifar10-vggvtiny-83.77-zxd.pth',
    'vggvvtiny': 'https://fake/models/cifar10-vggvvtiny-82.79-zxd.pth'
}


def cifar10_vggtiny_llsqs_bwns(pretrained=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    quan_type_act = model_name.split('_')[-2]
    quan_type_wt = model_name.split('_')[-1]
    quan_factory_act = my_nn.QuantizationFactory(quan_type_act, **kwargs)
    quan_factory_wt = my_nn.QuantizationFactory(quan_type_wt, **kwargs)
    model = _VGGQQ('VGG7', quan_factory_act, quan_factory_wt)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vgg7_ps'], map_location='cpu'),
                            '{}_map.json'.format(model_name))
    return model


# F represents that the first and last layers are also be quantized.
def cifar10_vggtiny_f_bwn(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Please use [dataset]_[architecture]_[quan_type] as the function name

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model_name = sys._getframe().f_code.co_name
    quan_type = model_name.split('_')[-1]
    quan_factory = my_nn.QuantizationFactory(quan_type, **kwargs)
    model = _VGGQF('VGG7', quan_factory)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vgg7'], map_location='cpu'),
                            '{}_map.json'.format(model_name))
    return model


def cifar10_vggtiny_bwn(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Please use [dataset]_[architecture]_[quan_type] as the function name

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model_name = sys._getframe().f_code.co_name
    quan_type = model_name.split('_')[-1]
    quan_factory = my_nn.QuantizationFactory(quan_type, **kwargs)
    model = _VGGQ('VGG7', quan_factory)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vgg7']),
                            '{}_map.json'.format(model_name))
    return model


def cifar10_vggtiny(pretrained=False, **kwargs):
    # The accuracy curve is not good enough.
    """VGG small model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('VGG7')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg7_ps']))
    return model


def cifar10_vggvtiny_llsqs_bwns(pretrained=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    quan_type_act = model_name.split('_')[-2]
    quan_type_wt = model_name.split('_')[-1]
    quan_factory_act = my_nn.QuantizationFactory(quan_type_act, **kwargs)
    quan_factory_wt = my_nn.QuantizationFactory(quan_type_wt, **kwargs)
    model = _VGGQQ('VGGvtiny', quan_factory_act, quan_factory_wt)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vggvtiny'], map_location='cpu'),
                            '{}_map.json'.format(model_name))
    return model


def cifar10_vggvtiny(pretrained=False, **kwargs):
    # The accuracy curve is not good enough.
    """VGG small model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('VGGvtiny')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vggvtiny']))
    return model


def cifar10_vggvvtiny(pretrained=False, **kwargs):
    # The accuracy curve is not good enough.
    """VGG small model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('VGGvvtiny')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vggvvtiny']))
    return model


def cifar10_vggvvtiny_llsqs_bwns(pretrained=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    quan_type_act = model_name.split('_')[-2]
    quan_type_wt = model_name.split('_')[-1]
    quan_factory_act = my_nn.QuantizationFactory(quan_type_act, **kwargs)
    quan_factory_wt = my_nn.QuantizationFactory(quan_type_wt, **kwargs)
    model = _VGGQQ('VGGvvtiny', quan_factory_act, quan_factory_wt)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vggvvtiny'], map_location='cpu'),
                            '{}_map.json'.format(model_name))
    return model


def cifar10_vggtiny_bn(pretrained=False, **kwargs):
    """VGG small model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGBN('VGG7')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg7_ps']))
    return model


cfg = {
    'VGG7': [128, 128, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGGvtiny': [64, 64, 'M', 64, 64, 128, 128, 'M'],
    'VGGvvtiny': [32, 32, 'M', 64, 64, 128, 128, 'M']
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        if vgg_name == 'VGG7':
            self.classifier = nn.Linear(4096, 10)
            self.bias = True
            self.pad = 1
            # todo: bias False
        elif vgg_name == 'VGGvtiny':
            self.bias = False
            self.pad = 0
            self.classifier = nn.Sequential(
                nn.Linear(1152, 256, bias=self.bias),
                nn.Linear(256, 10, bias=self.bias))
        elif vgg_name == 'VGGvvtiny':
            self.bias = False
            self.pad = 0
            self.classifier = nn.Sequential(
                nn.Linear(1152, 256, bias=self.bias),
                nn.Linear(256, 10, bias=self.bias))
        self.features = self._make_layers(cfg[vgg_name])
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=self.pad, bias=self.bias),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class VGGBN(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBN, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(4096, 10)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class _VGGQ(nn.Module):
    def __init__(self, vgg_name, quan_factory):
        super(_VGGQ, self).__init__()
        self.quan_factory = quan_factory
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(4096, 10)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # first layer
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
            else:
                layers += [
                    self.quan_factory.product_Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class _VGGQF(nn.Module):
    def __init__(self, vgg_name, quan_factory):
        super(_VGGQF, self).__init__()
        self.quan_factory = quan_factory
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self.quan_factory.product_LinearQ(4096, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  # first layer
                layers += [
                    self.quan_factory.product_Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class _VGGQQ(nn.Module):
    def __init__(self, vgg_name, quan_factory_act, quan_factory_wt):
        super(_VGGQQ, self).__init__()
        self.quan_factory_act = quan_factory_act
        self.quan_factory_wt = quan_factory_wt

        if vgg_name == 'VGG7':
            self.classifier = nn.Sequential(
                self.quan_factory_act.product_ActQ(),
                self.quan_factory_wt.product_LinearQ(4096, 10))
            self.bias = True
            self.pad = 1
            # todo: bias False
        elif vgg_name == 'VGGvtiny' or 'VGGvvtiny':
            self.bias = False
            self.pad = 0
            self.classifier = nn.Sequential(
                nn.Linear(1152, 256, bias=self.bias),
                nn.Linear(256, 10, bias=self.bias))
            self.classifier = nn.Sequential(
                self.quan_factory_act.product_ActQ(),
                self.quan_factory_wt.product_LinearQ(1152, 256, bias=self.bias),
                self.quan_factory_act.product_ActQ(),
                self.quan_factory_wt.product_LinearQ(256, 10, bias=self.bias)
            )

        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  # first layer
                layers += [
                    self.quan_factory_act.product_ActQ(),
                    self.quan_factory_wt.product_Conv2dQ(in_channels, x, kernel_size=3, padding=self.pad,
                                                         bias=self.bias),
                    # nn.Dropout2d(p=0.5),
                    nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

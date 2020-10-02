import sys

import torch.utils.model_zoo as model_zoo

import models._modules as my_nn
from models import load_pre_state_dict
from models.cifar10 import _VGGQ

__all__ = [
    'cifar10_vggsmall_dnq', 'cifar10_vggsmall_dnqv2'
]

# model name: [dataset]-[architecture]-[acc]-zxd.pth
model_urls = {
    '----': 'https://fake/models/cifar10-vgg7-zxd-8943fa3.pth',
}
cfg = {
    'VGG7': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG7Q': [128, 128, 'M', 256, 256, 'M', 512],
}


def cifar10_vggsmall_dnq(pretrained=False, **kwargs):
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
        assert NotImplementedError
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vgg7']),
                            '{}_map.json'.format(model_name))
    return model


def cifar10_vggsmall_dnqv2(pretrained=False, **kwargs):
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
        assert NotImplementedError
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vgg7']),
                            '{}_map.json'.format(model_name))
    return model

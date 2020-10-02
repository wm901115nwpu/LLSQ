'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from models._modules import Conv2dClusterQ

__all__ = [
    'VGG_Cluster', 'cifar10_vggsmall_cluster_q'
]

model_urls = {
    'vgg7': 'https://fake/models/cifar10-vgg7-zxd-8943fa3.pth',
}


def cifar10_vggsmall_cluster_q(pretrained=False, **kwargs):
    """VGG small model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_Cluster('VGG7', nbits_w=kwargs['nbits_w'])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg7']), strict=False)
    return model


cfg = {
    'VGG7': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
}


class VGG_Cluster(nn.Module):
    def __init__(self, vgg_name, nbits_w):
        super(VGG_Cluster, self).__init__()
        self.nbits_w = nbits_w
        self.features = self._make_layers(cfg[vgg_name])
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Linear(512 * scale, 10)
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
                layers += [Conv2dClusterQ(in_channels, x, kernel_size=3, padding=1, bias=False, nbits=self.nbits_w),
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

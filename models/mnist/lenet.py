'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import models._modules as my_nn

__all__ = ['LeNet', 'mnist_lenet', 'mnist_lenet_tbq']
model_urls = {
    'lenet': 'https://fake/models/mnist-lenet-xxx-xxx.pth'
}


def mnist_lenet_tbq(pretrained=False, **kwargs):
    model = LeNetTBQ()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['lenet'], map_location='cpu'))
    return model


def mnist_lenet(pretrained=False, **kwargs):
    model = LeNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['lenet'], map_location='cpu'))
    return model


class LeNetTBQ(nn.Module):
    def __init__(self):
        super(LeNetTBQ, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = my_nn.Conv2dTBQ(6, 16, 5)
        self.fc1 = my_nn.LinearTBQ(16 * 5 * 5, 120)
        self.fc2 = my_nn.LinearTBQ(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

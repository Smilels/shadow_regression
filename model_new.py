import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class SimpleRegression(nn.Module):
    def __init__(self):
        super(SimpleRegression, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.pos_output = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
        )

    def forward(self, fore_bf):
        fb_feature = self.resnet(fore_bf) # torch.Size :512*1*1
        fb_feature = fb_feature.view(-1,512)
        # print(fb_feature.shape)
        pos_feature = self.pos_output(fb_feature) # torch.Size([1, 22])

        return pos_feature


if __name__ == '__main__':
    data = Variable(torch.ones([1, 3, 224, 224]))
    data[0,0,0,1] = 10
    data[0,1,1,1] = 10
    data[0,2,1,2] = 10
    # layer = SpatialSoftmax(3, 3, 3, temperature=1)
    model = SimpleRegression()
    modules = list(model.children())
    # print(modules[0].spatialsoftmaxlayer.parameters)
    # print(model(data)[1])
    #  a=0
    # for k in model.children():
    #     a=a+1
    #     # print(k)
    #     if a>1:
    #         print(list(k.children())[0])
    #
    #         # torch.nn.init.kaiming_uniform_(list(k.children())[0].weight, a=0, mode='fan_in')
    #         # torch.nn.init.constant(list(k.children())[0].bias, 0.1)
    #         print(list(k.children())[0].weight)
    #         print(list(k.children())[0].bias)




import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        #生成x轴是-1,1 分成self.height块的网格轴，pos_x and pos_y  size are self.width * self.height
        # each line in pos_x is same, each column in pos_y is same
        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        # print(pos_x)
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # data_format 设置为 "NHWC" 时，排列顺序为 [batch, height, width, channels]；
        # 设置为 "NCHW" 时，排列顺序为 [batch, channels, height, width]。
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x)*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y)*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class ResNetEmbedSpatial(nn.Module):
    def __init__(self, spatial_channel=128):
        super(ResNetEmbedSpatial, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-4]
        self.resnet = nn.Sequential(*modules)
        self.spatialsoftmaxlayer = SpatialSoftmax(28, 28, spatial_channel, temperature=1)  # the resnet18 conv feature:512*1*1

    def forward(self, x):
        feature = self.resnet(x)
        feature = self.spatialsoftmaxlayer(feature)
        return feature

# tcn
class Tripletnet(nn.Module):
    def __init__(self):
        super(Tripletnet, self).__init__()
        self.spa_c = 128
        self.feature = ResNetEmbedSpatial(spatial_channel=self.spa_c)

        self.fc1 = nn.Linear(self.feature.view(1,-1).size(1), 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, fore_bf, left_bf, fore_af):
        fb_feature = self.feature(fore_bf)
        lb_feature = self.feature(left_bf)
        fr_feature = self.feature(fore_af)
        fb_feature = self.fc1(fb_feature)
        lb_feature = self.fc1(lb_feature)
        fr_feature = self.fc1(fr_feature)
        fb_feature = self.fc2(fb_feature)
        lb_feature = self.fc2(lb_feature)
        fr_feature = self.fc2(fr_feature)
        dist_a = F.pairwise_distance(fb_feature, lb_feature, 2)
        dist_b = F.pairwise_distance(fb_feature, fr_feature, 2)

        return dist_a, dist_b, fb_feature, lb_feature, fr_feature

# two camera
class JointsResNetEmbedSpatial(nn.Module):
    def __init__(self):
        super(JointsResNetEmbedSpatial, self).__init__()
        self.spa_c = 128
        self.feature = ResNetEmbedSpatial(spatial_channel=self.spa_c,pretrained=True)
        # for param in self.feature.parameters():
        #      param.requires_grad = False

        self.pos_output = nn.Sequential(
            nn.Linear(2*self.spa_c*4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 24)
        )

    def forward(self, fore_bf, left_bf):
        fb_feature = self.feature(fore_bf)
        lb_feature = self.feature(left_bf)
        embedding = torch.cat((fb_feature, lb_feature), 1)
        pos_feature = self.pos_output(embedding)

        return pos_feature, fb_feature, lb_feature

# one camera
class SimpleRegression(nn.Module):
    def __init__(self):
        super(SimpleRegression, self).__init__()
        self.spa_c = 128
        self.feature = ResNetEmbedSpatial(spatial_channel=self.spa_c)
        for param in self.feature.parameters():
             param.requires_grad = True

        self.pos_output = nn.Sequential(
            nn.Linear(2*self.spa_c, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
        )

    def forward(self, fore_bf):
        fb_feature = self.feature(fore_bf) # torch.Size([1, 24])
        pos_feature = self.pos_output(fb_feature) # torch.Size([1, 256])

        return pos_feature, fb_feature


if __name__ == '__main__':
    from torch.autograd import Variable
    data = Variable(torch.ones([1, 3, 224, 224]))
    data[0,0,0,1] = 10
    data[0,1,1,1] = 10
    data[0,2,1,2] = 10
    # layer = SpatialSoftmax(3, 3, 3, temperature=1)
    model = SimpleRegression()
    #print(model(data))
    # print(model(data)[0])
    # print(model(data)[1])



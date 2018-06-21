import torch.nn as nn
import torch.nn.functional as F
import torch

class Self_supervised_network(nn.Module):
    def __init__(self):
        super(Self_supervised_network, self).__init__()
        self.img_h = 128
        self.img_w = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5),
            torch.nn.init.xavier_uniform_(self.conv1.weight),
            torch.nn.init.constant_(self.conv1.bias, 0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(8, 8, kernel_size=5),
            torch.nn.init.xavier_uniform_(self.conv2.weight),
            torch.nn.init.constant_(self.conv2.bias, 0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3),
            torch.nn.init.xavier_uniform_(self.conv3.weight),
            torch.nn.init.constant_(self.conv3.bias, 0.1),
            nn.ReLU()
        )

        self.latent_layer = nn.Sequential(
            nn.Linear(8 * 23 * 23, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.latent_output = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8 * 23 * 23)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3),
            torch.nn.init.xavier_uniform_(self.conv3.weight),
            torch.nn.init.constant_(self.conv3.bias, 0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(8, 8, kernel_size=5),
            torch.nn.init.xavier_uniform_(self.conv2.weight),
            torch.nn.init.constant_(self.conv2.bias, 0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.ConvTranspose2d(8, 3, kernel_size=5),
            torch.nn.init.xavier_uniform_(self.conv1.weight),
            torch.nn.init.constant_(self.conv1.bias, 0.1),
            nn.ReLU()
        )

    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

    def forward(self, image):
        encoder_feature = self.encoder(image) # torch.Size :512*1*1
        encoder_feature = encoder_feature.view(-1,8 * 23 * 23)
        # print(fb_feature.shape)
        latent_feature = self.latent_layer(encoder_feature) # torch.Size([1, 22])
        latent_output = self.latent_output(latent_feature)
        latent_output = latent_output.view(8,23,23)
        image_out = self.decoder(latent_output)
        return latent_feature, image_out

# dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)

class Joint_net(nn.Module):
    def __init__(self):
        super(Joint_net, self).__init__()
        self.feature = Self_supervised_network()
        self.fc1 = nn.Linear(8 * 23 * 23, 1024)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        self.fc_relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 1024)
        torch.nn.init.normal_(self.fc2.weight, std=0.001)
        torch.nn.init.constant_(self.fc2.bias, 0.1)
        self.fc_relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, 22)
        torch.nn.init.normal_(self.fc3.weight, std=0.001)
        torch.nn.init.constant_(self.fc3.bias, 0.1)

    def forward(self, image):
        feature = self.feature(image)
        feature = feature.view(-1, 8 * 23 * 23)
        fc1 = self.fc1(feature)
        fc_relu1 = self.fc_relu1(fc1)
        dropout1 = self.dropout1(fc_relu1)
        fc2 = self.fc2(dropout1)
        fc_relu2 = self.fc_relu2(fc2)
        dropout2 = self.dropout2(fc_relu2)
        pos_feature = self.fc3(dropout2)


if __name__ == '__main__':
    model = Self_supervised_network(22)
    # params=model.state_dict()
    # modules = list(model.children())
    # print (list(modules[-1].children())[-1].weight.view(-1,1)[-1])
    # print (list(modules[-1].children())[-1].weight.view(-1,1))

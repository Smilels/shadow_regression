import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

#def backward_hook(module, grad_in, grad_out):
    # print(module)
    # print('grad of output:', grad_out)
    # print('grad of input:', grad_in)

#def forward_hook(module, input, output):
#    print(module)
#    print('input:', input)
#    print('output:', output)

class Priornet(nn.Module):
    def __init__(self):
        super(Priornet, self).__init__()
        self.img_h = 224
        self.img_w = 224
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.constant_(self.conv1.bias,0.1)
        # self.conv1.register_forward_hook(forward_hook)
	# self.conv1.register_backward_hook(backward_hook)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.constant_(self.conv2.bias,0.1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.constant_(self.conv3.bias, 0.1)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(8 * 23 * 23, 1024)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        self.fc_relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 1024)
        torch.nn.init.normal_(self.fc2.weight,std=0.001)
        torch.nn.init.constant_(self.fc2.bias, 0.1)
        self.fc_relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, 22)
        torch.nn.init.normal_(self.fc3.weight, std=0.001)
        torch.nn.init.constant_(self.fc3.bias, 0.1)

    def forward(self, image):
        conv1 = self.conv1(image)
        relu1 = self.relu1(conv1)
        pool1 = self.pool1(relu1)
        conv2 = self.conv2(pool1)
        relu2 = self.relu2(conv2)
        pool2 = self.pool2(relu2)
        conv3 = self.conv3(pool2)
        relu3 = self.relu3(conv3)
        # print(relu3.shape)
        relu3 = relu3.view(-1,8 * 23 * 23)
        fc1 = self.fc1(relu3)
        fc_relu1 = self.fc_relu1(fc1)
        dropout1 = self.dropout1(fc_relu1)
        fc2 = self.fc2(dropout1)
        fc_relu2 = self.fc_relu2(fc2)
        dropout2 = self.dropout2(fc_relu2)
        pos_feature = self.fc3(dropout2)

        return pos_feature

if __name__ == '__main__':
    data = Variable(torch.ones([1,3, 224, 224]))
    model = Priornet()
    a = model(data)
    modules = list(model.children())

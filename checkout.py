from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from model import SimpleRegression
from triplet_image_loader import SimpleImageLoader
from visdom import Visdom
import numpy as np
import torch.utils.model_zoo as model_zoo


_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='checkoutpoint/Shadow_imitation_gpu/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Shadow_imitation_gpu', type=str,
                    help='name of experiment')
parser.add_argument('--net', default='SIMPLE', type=str,
                    help='name of Trainning net')
best_acc = 0

def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 20, 'pin_memory': True} if args.cuda else {}

    print('==>Preparing data...')

    base_path = "./data/handpose_data/"
    train_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(base_path, train=True,
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                transforms.ToTensor(),
                                Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                            ])),
        batch_size=args.batch_size, shuffle=True, drop_last = False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(base_path, False,
                        transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                transforms.ToTensor(),
                                Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])),
        batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)

    jnet = SimpleRegression()
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            if isinstance(jnet, nn.DataParallel):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    namekey = k[7:]  # remove `module.  TODO： check the meaning of 7 in here
                    new_state_dict[namekey] = v
                jnet.load_state_dict(new_state_dict)
            else:
                jnet.load_state_dict(checkpoint['state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

if __name__ == '__main__':
    main()
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
from model_new import SimpleRegression
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
parser.add_argument('--batch-size', type=int, default=168, metavar='N',
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
parser.add_argument('--resume', default='resume', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Shadow_imitation_gpu3', type=str,
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
        # torch.cuda.set_device(0)
    global plotter 
    plotter = VisdomLinePlotter(env_name=args.name)

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
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])),
        batch_size=args.batch_size, drop_last=False, **kwargs)

    jnet = SimpleRegression()

    # optionally resume from a checkpoint
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
                    namekey = k[7:]  # remove module.  TODO： check the meaning of 7 in here
                    new_state_dict[namekey] = v
                jnet.load_state_dict(new_state_dict)
            else:
                jnet.load_state_dict(checkpoint['state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        jnet.cuda()
        if torch.cuda.device_count() > 1:
           jnet = nn.DataParallel(jnet,device_ids=[0,1])  # 使用dataParallel重新包装一下

    # This flag allows you to enable the inbuilt cudnn auto-tuner to
    # find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(jnet.parameters(), lr=args.lr, momentum=args.momentum)
    if args.cuda and torch.cuda.device_count() > 1:
        optimizer = nn.DataParallel(optimizer,device_ids=[0,1])

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, jnet, criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(test_loader, jnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if isinstance(jnet, nn.DataParallel):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': jnet.module.state_dict(),
                'best_prec1': best_acc,
            }, is_best)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': jnet.state_dict(),
                'best_prec1': best_acc,
            }, is_best)

        # state_dict() Returns a dictionary containing a whole state of the module.
        # Both parameters and persistent buffers (e.g. running averages) are included.
        # Keys are corresponding parameter and buffer names.


def train(train_loader, jnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs1 = AverageMeter()
    accs2 = AverageMeter()
    accs3 = AverageMeter()

    # switch to train mode
    jnet.train()
    for batch_idx, (data1, joint_target) in enumerate(train_loader):
        joint_target = torch.t(torch.stack(joint_target)).float()
        if args.cuda:
            data1, joint_target = data1.cuda(), joint_target.cuda()

        data1, joint_target = Variable(data1), Variable(joint_target)

        # compute output
        pos_feature = jnet(data1)
        loss = criterion(pos_feature, joint_target)

        # adjust_learning_rate(optimizer, epoch)
        optimizer.zero_grad()
        loss.backward()
        if isinstance(jnet, nn.DataParallel):
            optimizer.module.step()
        else:
            optimizer.step()

        acc = accuracy(pos_feature, joint_target, accuracy_thre=[0.05, 0.1, 0.2])
        # error solution "TypeError: tensor(0.5809) is not JSON serializable"
        ll = loss.data
        losses.update(ll, data1.size(0))
        accs1.update(acc[0], data1.size(0))
        accs2.update(acc[1], data1.size(0))
        accs3.update(acc[2], data1.size(0))

        if batch_idx % args.log_interval == 0:
            print('Train Simple Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc1: {:.2f}% ({:.2f}%) \t'
                  'Acc2: {:.2f}% ({:.2f}%) \t'
                  'Acc3: {:.2f}% ({:.2f}%) '.format(
                    epoch, batch_idx * len(data1), len(train_loader.dataset),
                    losses.val, losses.avg, 100. * accs1.val, 100. * accs1.avg,
                    100. * accs2.val, 100. * accs2.avg,
                    100. * accs3.val, 100. * accs3.avg))

    # log avg values to somewhere
    plotter.plot('acc1', 'train', epoch, accs1.avg)
    plotter.plot('acc2', 'train', epoch, accs2.avg)
    plotter.plot('acc3', 'train', epoch, accs3.avg)
    plotter.plot('loss', 'train', epoch, losses.avg)


def test(test_loader, jnet, criterion, epoch):
    correct1 = 0.0
    correct2 = 0.0
    correct3 = 0.0
    total = 0
    losses = AverageMeter()
    accs1 = AverageMeter()
    accs2 = AverageMeter()
    accs3 = AverageMeter()
    # switch to evaluation mode
    jnet.eval()

    for batch_idx, (data1, joint_target) in enumerate(test_loader):
        joint_target = torch.t(torch.stack(joint_target)).float()
        if args.cuda:
            data1, joint_target = data1.cuda(), joint_target.cuda()

        data1, joint_target = Variable(data1), Variable(joint_target)

        # compute output
        pos_feature = jnet(data1)
        loss = criterion(pos_feature, joint_target)

        acc = accuracy(pos_feature, joint_target, accuracy_thre=[0.05, 0.1, 0.2])
        ll = loss.data
        losses.update(ll, data1.size(0))
        accs1.update(acc[0], data1.size(0))
        accs2.update(acc[1], data1.size(0))
        accs3.update(acc[2], data1.size(0))

        if batch_idx % args.log_interval == 0:
            print('Test Simple Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc1: {:.2f}% ({:.2f}%) \t'
                  'Acc2: {:.2f}% ({:.2f}%) \t'
                  'Acc3: {:.2f}% ({:.2f}%)'.format(
                    epoch, batch_idx * len(data1), len(test_loader.dataset),
                    losses.val, losses.avg,
                    100. * accs1.val, 100. * accs1.avg, 100. * accs2.val,
                    100. * accs2.avg, 100. * accs3.val, 100. * accs3.avg))

    # log avg values to somewhere
    plotter.plot('acc1', 'test', epoch, accs1.avg)
    plotter.plot('acc2', 'test', epoch, accs2.avg)
    plotter.plot('acc3', 'test', epoch, accs3.avg)
    plotter.plot('loss', 'test', epoch, losses.avg)

    return accs3.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "checkoutpoint/%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkoutpoint/%s/' % args.name + 'model_best.pth.tar')


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]),
                env=self.env, name=split_name, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), win = self.plots[var_name], env=self.env,
                        name=split_name, update='append')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 50 epochs"""
    lr = args.lr * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        # for param_group in optimizer.module.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, accuracy_thre):
    """Computes the precision@k for the specified values of k"""
    acc=[]
    total = target.size(0)
    dist = (output - target).abs().max(1)[0]
    for k in accuracy_thre:
        correct = 0
        for i in dist:
            if i < k:
                correct += 1
        acc.append(correct/total)
    return acc

if __name__ == '__main__':
    main()    

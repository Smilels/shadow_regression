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
parser.add_argument('--batch-size', type=int, default=120, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=600, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
parser.add_argument('--name', default='Shadow_imitation_gpu4', type=str,
                    help='name of experiment')
parser.add_argument('--net', default='SIMPLE', type=str,
                    help='name of Trainning net')
parser.add_argument('--parallel', action='store_true',default=True,
                    help='enables dataparallel')
best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(3)
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
    if args.cuda:
        jnet.cuda()
        if torch.cuda.device_count() > 1 and args.parallel:
           jnet = nn.DataParallel(jnet)  # dataParallel

    # This flag allows you to enable the inbuilt cudnn auto-tuner to
    # find the best algorithm to use for your hardware.
    cudnn.benchmark = True

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
                    namekey = k[7:]  # remove module.  TODO: check the meaning of 7 in here
                    new_state_dict[namekey] = v
                jnet.load_state_dict(new_state_dict)
            else:
                jnet.load_state_dict(checkpoint['state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(jnet.parameters(), lr=args.lr, momentum=args.momentum)
    if isinstance(jnet, nn.DataParallel):
        optimizer = nn.DataParallel(optimizer,device_ids=[0,1])

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        adjust_learning_rate(jnet, optimizer, epoch)
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
    correct1 = 0.0
    correct2 = 0.0
    correct3 = 0.0
    total = 0
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
        # print("joint target is ",joint_target)
        pos_feature = jnet(data1)
        loss_joint = criterion(pos_feature, joint_target)
        # print("loss_joint is ",loss_joint)
        # print("pos_feature is ", pos_feature)
        loss_cons = joint_constraits(pos_feature)
        #print("loss_cons is", loss_cons)
        loss = 10 * loss_joint + loss_cons
       # print("loss is ",loss)
        optimizer.zero_grad()
        loss.backward()
        #g = make_dot(loss)
        #g.view()
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
                  'Loss: {:.4f} ({:.4f}) {:.4f}\t'
                  'Acc1: {:.2f}% ({:.2f}%) \t'
                  'Acc2: {:.2f}% ({:.2f}%) \t'
                  'Acc3: {:.2f}% ({:.2f}%) '.format(
                    epoch, batch_idx * len(data1), len(train_loader.dataset),
                    losses.val, losses.avg, loss_cons, 100. * accs1.val, 100. * accs1.avg,
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
        # print("joint target is ",joint_target)
        pos_feature = jnet(data1)
        loss_joint = criterion(pos_feature, joint_target)
        # print("loss_joint is ",loss_joint)
        loss_cons = joint_constraits(pos_feature)
        # print("loss_cons is", loss_cons)
        loss = loss_joint + 0.1 * loss_cons
        # print("loss is ",loss)

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


def joint_constraits(pos_feature):
    F4 = [pos_feature[:, 3], pos_feature[:, 7], pos_feature[:, 12], pos_feature[:, 16]]
    F1_3 = [pos_feature[:, 1], pos_feature[:, 5], pos_feature[:, 9], pos_feature[:, 14],
            pos_feature[:, 2], pos_feature[:, 6], pos_feature[:, 10], pos_feature[:, 15],
            pos_feature[:, 0], pos_feature[:, 4], pos_feature[:, 11], pos_feature[:, 13],
            pos_feature[:, 17]]
    loss_cons = 0

    for pos in F1_3:
        for f in pos:
            loss_cons = loss_cons + max(0 - f, 0) + max(f - 1.57, 0)
    for pos in F4:
        for f in pos:
            loss_cons = loss_cons + max(-0.349 - f, 0) + max(f - 0.349, 0)
    for f in pos_feature[:, 8]:
        loss_cons = loss_cons + max(0 - f, 0) + max(f - 0.785, 0)
    for f in pos_feature[:, 18]:
        loss_cons = loss_cons + max(-0.524 - f, 0) + max(f - 0.524, 0)
    for f in pos_feature[:, 19]:
        loss_cons = loss_cons + max(-0.209 - f, 0) + max(f - 0.209, 0)
    for f in pos_feature[:, 20]:
        loss_cons = loss_cons + max(0 - f, 0) + max(f - 1.222, 0)
    for f in pos_feature[:, 21]:
        loss_cons = loss_cons + max(-1.047 - f, 0) + max(f - 1.047, 0)
    return loss_cons


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


def adjust_learning_rate(jnet, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 50 epochs"""
    lr = args.lr * (0.5 ** (epoch // 50))
    print("current laerning rate is ", lr)
    if isinstance(jnet, nn.DataParallel):
       for param_group in optimizer.module.param_groups:
          param_group['lr'] = lr
    else:
       for param_group in optimizer.param_groups:
          # for param_group in optimizer.module.param_groups:
          param_group['lr'] = lr



def accuracy(output, target, accuracy_thre):
    """Computes the precision@k for the specified values of k"""
    acc=[]
    total = target.size(0)
    dist = (output - target).abs().max(1)[0]
    #print("dist is", dist)
    for k in accuracy_thre:
        correct = 0
        for i in dist:
            if i < k:
                correct += 1
        acc.append(correct/total)
    return acc


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


if __name__ == '__main__':
    main()    

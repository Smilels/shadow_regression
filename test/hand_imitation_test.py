from __future__ import print_function
import argparse
import sys
import os
import rospy
import moveit_commander
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from model_resnet import SimpleRegression
from triplet_image_loader_test import  TestImageLoader
import numpy as np

import torch.utils.model_zoo as model_zoo

'''print group.get_joint_value_target() joints order:
  [rh_WRJ2, rh_WRJ1, rh_FFJ4, rh_FFJ3, 
  rh_FFJ2, rh_FFJ1, rh_LFJ5, rh_LFJ4,
  rh_LFJ3, rh_LFJ2, rh_LFJ1, rh_MFJ4, 
  rh_MFJ3, rh_MFJ2, rh_MFJ1, rh_RFJ4,
  rh_RFJ3, rh_RFJ2, rh_RFJ1, rh_THJ5,
  rh_THJ4, rh_THJ3, rh_THJ2, rh_THJ1]'''
'''dataset joints order:
[rh_FFJ1, rh_FFJ2, rh_FFJ3, rh_FFJ4, 
 rh_LFJ1, rh_LFJ2, rh_LFJ3, rh_LFJ4,
 rh_LFJ5, rh_MFJ1, rh_MFJ2, rh_MFJ3,
 rh_MFJ4, rh_RFJ1, rh_RFJ2, rh_RFJ3,
 rh_RFJ4, rh_THJ1, rh_THJ2, rh_THJ3,
 rh_THJ4, rh_THJ5, rh_WRJ1, rh_WRJ2]
'''
current_variable_values[0], current_variable_values[1], estimate_values[3], estimate_values[2],
estimate_values[1], estimate_values[0], estimate_values[8], estimate_values[7], \
estimate_values[6], estimate_values[5], estimate_values[4], estimate_values[12],
estimate_values[11], estimate_values[10], estimate_values[9], estimate_values[16],
estimate_values[15], estimate_values[14], estimate_values[13], estimate_values[21],
estimate_values[20], estimate_values[19], estimate_values[18], estimate_values[17]
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--model', default='resume', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--parallel', action='store_true',default=False,
                    help='enables Dataparallel')

def main():
    global args
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(0)
    global plotter

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial',
                    anonymous=True)
    group = moveit_commander.MoveGroupCommander("right_hand")
    group.set_named_target("open")
    group.go()
    rospy.sleep(5)

    current_joints = group.get_current_joint_values()

    kwargs = {'num_workers': 20, 'pin_memory': True} if args.cuda else {}
    print('==>Preparing data...')
    base_path = "../data/handpose_data_test/"
    test_loader = torch.utils.data.DataLoader(
        TestImageLoader(base_path, 20,
                        transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                            ])),
        batch_size=args.batch_size, **kwargs)

    jnet = SimpleRegression()
    if args.cuda:
        jnet.cuda()
        if torch.cuda.device_count() > 1 and args.parallel:
           jnet = nn.DataParallel(jnet,device_ids=[0,1])  # dataParallel

    cudnn.benchmark = True
    criterion = torch.nn.MSELoss()
    # optionally resume from a checkpoint
    if args.model:
        if os.path.isfile(args.model):
            print("==> loading model '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            jnet.load_state_dict(checkpoint['state_dict'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==>ERROR: no model found at '{}'".format(args.model))

    while not rospy.is_shutdown():
            estimate_joints = Get_Estimation_Joints(jnet, criterion, test_loader)
            target_joints = [current_joints[0], current_joints[1], estimate_joints[3], estimate_joints[2],
            estimate_joints[1], estimate_joints[0], estimate_joints[8], estimate_joints[7], \
            estimate_joints[6], estimate_joints[5], estimate_joints[4], estimate_joints[12],
            estimate_joints[11], estimate_joints[10], estimate_joints[9], estimate_joints[16],
            estimate_joints[15], estimate_joints[14], estimate_joints[13], estimate_joints[21],
            estimate_joints[20], estimate_joints[19], estimate_joints[18], estimate_joints[17]]

            group.set_joint_value_target(target_joints)
            group.go()
            rospy.sleep(5)


def Get_Estimation_Joints(jnet, criterion, test_loader):
    for batch_idx, (data, joint_target) in enumerate(test_loader):

        data_ = torch.squeeze(data)
        to_pil_image = transforms.ToPILImage()
        img = to_pil_image(data_)
        img.show()

        joint_target = torch.t(torch.stack(joint_target)).float()
        if args.cuda:
            data, joint_target = data.cuda(), joint_target.cuda()

        data, joint_target = Variable(data), Variable(joint_target)

        # compute output
        estimate_joints = jnet(data)
        loss_joint = criterion(estimate_joints, joint_target)
        loss_cons = joint_constraits(estimate_joints)
        loss = 10 * loss_joint + loss_cons
        acc = accuracy(estimate_joints, joint_target, accuracy_thre=[0.05, 0.1, 0.2])

        print('Test Image Epoch:\t'
              'Loss: {:.4f} ({:.4f}) ({:.4f}) \t'
              'Acc Num: {:d} {:d} {:d}'.format(
               loss, loss_joint, loss_cons,
               acc[0], acc[1], acc[2]))

        return estimate_joints


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
        acc.append(correct)
    return acc


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


if __name__ == '__main__':
    main()
'''
pairs of human hand images and shadow hand images loaders
human hand depth images: form bighand2.2 dataset and crop by ground truth of keypoints annotation
shadow depth images: take pictures by shadow and use mapping method to go to similar pose like human
'''
import os.path
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from torch.autograd import Variable


class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, train=True, transform=None):
        """ train_images: pairs of human hand images and shadow hand images """
        self.base_path = base_path
        DataFile = open(self.base_path + "test.txt", "r")

        lines = DataFile.read().splitlines()
        self.framelist = [ln.split(' ')[0].replace("\t", "") for ln in lines]
        label_source = [ln.split('\t')[1:] for ln in lines]
        self.label = []
        for ln in label_source:
            ll = ln[0:63]
            self.label.append([float(l.replace(" ", "")) for l in ll])

        self.label = np.array(self.label)
        DataFile.close()
        self.num_data = len(self.framelist)
        self.transform = transform

    def __getitem__(self, index):
        idx = self.framelist[index]
        h_img = Image.open(self.base_path + str(idx)) # 640*320
        h_img.show()

        keypoints = self.label[index]
        keypoints = keypoints.reshape(21, 3)

        # camera center coordinates and focal length
        mat = np.array([[475.065948,0, 315.944855],[0,475.065857,245.287079],[0,0,1]])
        uv = np.random.randn(21,2)

        for i in range(0,len(keypoints)):
            uv[i] = ((1/keypoints[i][2]) * mat @ keypoints[i])[0:2]

        # from IPython import embed;embed()

        # Image coordinates: origin at the top-left corner, u axis going right and v axis going down
        padding = 10
        left = uv.min(axis = 0)[0]
        top = uv.min(axis=0)[1]
        right = uv.max(axis=0)[0]
        bottom = uv.max(axis=0)[1]
        h_img = h_img.crop((left-padding, top-padding, right+padding, bottom+padding))
        # width, height = h_img.size  # Get dimensions
        # print(width)
        # print(height)
        # h_img.show()

        # TODO:normalized to 96×96 pixels
        if self.transform is not None:
            h_img = self.transform(h_img)

        return h_img

    def __len__(self):
        return self.num_data

if __name__ == '__main__':
    base_path = "./data/trainning/"
    train = SimpleImageLoader(base_path,True,
                              transform=transforms.Compose([
                                 transforms.Resize(96),
                                 # transforms.CenterCrop(224),
                                 # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                  transforms.ToTensor(),
                                 # Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                                 #  transforms.Normalize(mean=[0.485, ], std=[0.229, ])
                              ]))
    h_img = train.__getitem__(5)

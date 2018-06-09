import os.path
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from numpy import random
from torch.autograd import Variable
import Augmentor


class TestImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, num = 20, transform=None):
        self.base_path = base_path
        if  os.path.isfile(self.base_path + "handpose_data_test_test.csv"):
            DataFile = open(self.base_path + "handpose_data_test_test.csv", "r")
            lines = DataFile.read().splitlines()

            test_lines = np.random.choice(lines, num)

            self.filenamelist = [ln.split(',')[0] for ln in test_lines]
            self.label = [ln.split(',')[1:] for ln in test_lines]
            # print(len(self.filenamelist))
            self.num_data = len(self.filenamelist)
            self.transform = transform
            DataFile.close()
        else:
            print('ERROR: can not find test data')

    def __getitem__(self, index):
        idx = self.filenamelist[index]
        img = Image.open("../" + str(idx))
        joints = [float(l.split(',')[0]) for l in self.label[index]]
        if self.transform is not None:
            img = self.transform(img)
        return img, joints

    def __len__(self):
        return self.num_data


if __name__ == '__main__':
    base_path = "../data/handpose_data_test/"
    test = TestImageLoader(base_path,2,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  # transforms.Normalize(mean=[0.485, ], std=[0.229, ])
                              ]))
    test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle=True, num_workers=2)

    # img,joint = test.__getitem__(1)
    # # print("image shape is", img.shape)
    # # print(joint)
    # to_pil_image = transforms.ToPILImage()
    # img = to_pil_image(img)
    # img.show()
    for batch_idx, (data, _) in enumerate(test_loader):

        data     = Variable(data).squeeze()
        data = torch.squeeze(data)
        to_pil_image = transforms.ToPILImage()
        # print(data.shape)
        img = to_pil_image(data)
        img.show()


import os.path
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from torch.autograd import Variable

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


class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, train=True, transform=None):
        """ train_label_file: a csv file with each image and correspongding label
            triplets_file_name: A text file with each line containing three integers,
                where integer i refers to the i-th image in the filenames file. """
        np.random.seed(42)
        a = np.arange(1,33)
        np.random.shuffle(a)
        self.base_path = base_path
        if not os.path.isfile(self.base_path + "handpose_data_train.csv") or not os.path.isfile(self.base_path + "handpose_data_test.csv"):
            csvSum = open(self.base_path + "handpose_data_train.csv", "w")
            writer = csv.writer(csvSum)
            csvSum_test = open(self.base_path + "handpose_data_test.csv", "w")
            writer_test = csv.writer(csvSum_test)
            for i in a[0:24]:
            # for i in range(1,2):
                 # os.path.join(
                each_path = self.base_path + "handpose_data" + str(i) + ".csv"
                csvFile = open(each_path , "r")
                reader = csv.reader(csvFile)

                for item in reader:
                    result = {}
                    column = './data/handpose' + str(i) + '/'+ str(item[0]) + '.jpg'
                    # result[column] = item[1:]
                    result = [column, item[1], item[2], item[3], item[4], item[5], item[6],
                              item[7], item[8], item[9], item[10], item[11], item[12], item[13],
                              item[14], item[15], item[16], item[17], item[18], item[19],
                              item[20], item[21], item[22]]
                    writer.writerow(result)
                csvFile.close()
            csvSum.close()

            for i in a[24:33]:
                 # os.path.join(
                each_path = self.base_path + "handpose_data" + str(i) + ".csv"
                csvFile = open(each_path , "r")
                reader = csv.reader(csvFile)

                for item in reader:
                    result = {}
                    column = 'data/handpose' + str(i) + '/'+ str(item[0]) + '.jpg'
                    # result[column] = item[1:]
                    result = [column, item[1], item[2], item[3], item[4], item[5], item[6],
                              item[7], item[8], item[9], item[10], item[11], item[12], item[13],
                              item[14], item[15], item[16], item[17], item[18], item[19],
                              item[20], item[21], item[22]]
                    writer_test.writerow(result)
                csvFile.close()
            csvSum_test.close()

        if train:
            DataFile = open(self.base_path + "handpose_data_train.csv", "r")
        else:
            DataFile = open(self.base_path + "handpose_data_test.csv", "r")

        lines = DataFile.read().splitlines()
        self.filenamelist = [ln.split(',')[0] for ln in lines]
        self.label = [ln.split(',')[1:] for ln in lines]
        # print(len(self.filenamelist))
        self.num_data = len(self.filenamelist)
        self.transform = transform
        DataFile.close()

    def __getitem__(self, index):
        idx = self.filenamelist[index]
        img = Image.open(str(idx))
        joints = []
        # for l in self.label[index]:
        #     print(l.split(','))
        #     joints.append(float(l.split(',')[0]))
        joints = [float(l.split(',')[0]) for l in self.label[index]]
        # joints = np.array(joints) # [24,1]
        if self.transform is not None:
            img = self.transform(img)

        return img, joints

    def __len__(self):
        return self.num_data

if __name__ == '__main__':
    base_path = "./data/handpose_data/"
    train = SimpleImageLoader(base_path,True,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                 # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                  transforms.ToTensor(),
                                 # Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                                  # transforms.Normalize(mean=[0.485, ], std=[0.229, ])
                              ]))
    train_loader = torch.utils.data.DataLoader(train, batch_size = train.__len__(), shuffle=True, num_workers=2)

    # img,joint = train.__getitem__(1330)
    # print("image shape is", img.shape)
    # print(joint)
    # to_pil_image = transforms.ToPILImage()
    # img = to_pil_image(img)
    # img.show()
    for step, (data,_) in enumerate(train_loader):
        data = Variable(data)
        print(step)
        data = data.numpy()
        # v1 = mean1 + data.float().mean()
        # std1 = std1 + data.float().std()
        #print(data.float().mean())
        #print(data.float().std())
        means = []
        stdevs = []
        for i in range(3):
            pixels = data[:,i,:,:].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))
        print("means: {}".format(means))
        print("stdevs: {}".format(stdevs))
        print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

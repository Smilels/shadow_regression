#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : pairs_image_loader.py
# Creation Date : 02-07-2018
# Created By : sli [sli@informatik.uni-hamburg.de]
import numpy as np
import shutil
import csv
import os

def main():
    a =np.ones(100,dtype=np.int64)
    # when generate human pose from annotation file open Training_Annotation.txt
    DataFile1 = open("../data/Training_Annotation.txt","r")
    lines1 = DataFile1.read().splitlines()
    # get corresponding genrated images from the num in random.csv
    DataFile = open("../data/random1.csv","r")
    lines = DataFile.read().splitlines()
    framelist = [lines1[int(ln)].split(' ')[0].replace("\t", "") for ln in lines]
    print(framelist)

    # random choose num from generated dataset imamge num
    # save random.csv use to human_robot_mappingfile.py generate human pose images
   # w_file = open("../data/random22.csv","w")
   # writer = csv.writer(w_file)
   # for x in range(100):
   #     t = np.random.randint(1,12000)
   #     a[x] = t
   #     writer.writerow([t])
   # w_file.close()

    # copy file
    destination_file = "../data/random_comparison/shadow_rgb/"
    for i in framelist:
        # print(a[i])
        source_file = "../data/rgb_shadow/" + i
        if os.path.isfile(source_file):
            shutil.copy(source_file,destination_file)

if __name__ == '__main__':
    main()

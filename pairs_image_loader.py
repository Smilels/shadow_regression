#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : pairs_image_loader.py
# Creation Date : 02-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]
import numpy as np
import shutil
import csv
import os
def main():
    a =np.ones(1000,dtype=np.int64)
    # DataFile = open("/media/robot/My Passport/data/training/Training_Annotation.txt","r")
    DataFile = open("./data/random.csv","r")
    lines = DataFile.read().splitlines()
    framelist = [ln.split(' ')[0].replace("\t", "") for ln in lines]
    print(len(framelist))


    # w_file = open("./data/random.csv","w")
    # writer = csv.writer(w_file)
    # for x in range(1000):
    #     t = np.random.randint(1,66585)
    #     # print(t)
    #     a[x] = t 
    #     writer.writerow([t])
    # w_file.close()

    destination_file = "/home/robot/workspace/shadow_hand/imitation/src/shadow_regression/data/choose/" 
    for i in range(1000):
        # print(a[i])
        source_file = "./data/rgb_shadow/" + str(int(framelist[i])+2) + ".png"
        if os.path.isfile(source_file):
            shutil.copy(source_file,destination_file)

if __name__ == '__main__':
    main()	

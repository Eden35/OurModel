#!/usr/bin/python3
# these code is for ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection
# -*- coding: utf-8 -*-
# @Author  : Duwei Dai

import os
import random
import numpy as np
from skimage import io
from PIL import Image

root_dir = '/data/TN3k'                # change it in your saved original data path
save_dir = './data/TN#K_npy_224_224'

if __name__ == '__main__':

    imgfile = os.path.join(root_dir, 'Images')
    all_files = os.listdir(imgfile)
    jpg_files = [f for f in all_files if f.endswith('.jpg')]

    print('Total files in directory:', len(all_files))
    print('Total .jpg files:', len(jpg_files))

    filename = sorted([os.path.join(imgfile, x) for x in jpg_files])
    # random.shuffle(filename)

    # Ensure the save directory exists
    image_save_dir = os.path.join(save_dir, 'images')
    if not os.path.isdir(image_save_dir):
        os.makedirs(image_save_dir)

    kk = 0
    for i in range(len(filename)):
        fname = os.path.splitext(os.path.basename(filename[i]))[0]

        image = Image.open(filename[i])
        image = image.resize((224, 224))
        image = np.array(image)

        images_img_filename = os.path.join(image_save_dir, fname + '.npy')
        np.save(images_img_filename, image)
        kk += 1
        print("Processed image number: ", kk)

    print('Successfully saved preprocessed data')

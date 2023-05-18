# Script to pre process bone marrow cell dataset
# name: data_augment.py
# author: mbwhiteh@sfu.ca
# date: 2022-03-29
import os
import skimage
import matplotlib.pyplot as plt
# files to augment
filename = "train_filenames.txt"
DATASET_PATH_ABS = os.path.abspath("BMC-Dataset")

with open(filename) as fd:
    for image_name in fd.readlines():
        image_name = image_name.strip("\n")
        if(image_name[0:3] == "NGB"):
            image_path = os.path.join(DATASET_PATH_ABS, image_name)
            img = skimage.io.imread(image_path)
            img = skimage.transform.rotate(img, 90)
            # img = skimage.filters.laplace(img)
            img = skimage.util.img_as_ubyte(img)
            new_img_name = "./Augmented_Data/" + image_name.strip(".jpg") + "_90Rotate.jpg"
            skimage.io.imsave(new_img_name, img)
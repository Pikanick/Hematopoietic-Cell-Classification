# Utility functions for dataset processing
# name: data_helper.py
# author: mbwhiteh@sfu.ca
# date: 2022-04-10

import os
import numpy as np
from tensorflow import keras
from PIL import Image
import math
import random

# These used to be os.path.abspath("../...") which resolves relative to
# the process's *current working directory*, not this file's location --
# so importing this module (from Model_Train.py / Model_Test.py, both run
# from the repo root) resolved "../train_filenames.txt" and
# "../BMC-Dataset" to one level *above* the repo, not the repo root. Anchor
# on this file's own location instead so it works regardless of where the
# caller's process was started from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH_ABS = os.path.join(_REPO_ROOT, "train_filenames.txt")
DATASET_PATH_ABS = os.path.join(_REPO_ROOT, "BMC-Dataset")
# Dataset label mappings for one-hot encoding
LABEL_DICT = {
    'BLA': 0,
    'LYT': 1,
    'NGB': 2,
    'NGS': 3
}

# Record metrics after each batch
class BatchLogger(keras.callbacks.Callback):
    def __init__(self, log_filename= None):
        self.log_filename = log_filename

    def on_epoch_end(self, batch, logs= None):
        with open(self.log_filename, 'a+') as fd:
            fd.write('{}\n'.format(logs))

# Data Generator class for batch training 
class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, X_set, Y_set, batch_size, num_classes):
        self.X_set = X_set
        self.Y_set = Y_set
        self.batch_size = batch_size
        self.class_count = num_classes

    # returns the length of the sequence
    def __len__(self):
        return math.ceil(len(self.X_set) / self.batch_size)
    
    # returns the batch of data and labels
    def __getitem__(self, idx):
        batch_of_filenames = self.X_set[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_of_labels = self.Y_set[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_X = list()
        # load batch of images
        for filepath in batch_of_filenames:
            abs_filepath = os.path.join(DATASET_PATH_ABS, filepath)
            try:
                img = keras.utils.load_img(abs_filepath, target_size=(128, 128))
                image_data = keras.utils.img_to_array(img)*(1/255)
                batch_X.append(image_data)
            except:
                print(f'File {abs_filepath} failed to load.')

        return (np.array([ img for img in batch_X]), keras.utils.to_categorical(batch_of_labels, num_classes= self.class_count))

# testing of Data Generator class
if __name__ == "__main__":
    # open training filenames
    with open(TRAIN_PATH_ABS) as train_fd:
        train_filenames = [f_name.strip('\n') for f_name in train_fd.readlines()]
        random.shuffle(train_filenames)
    # associated labels mapped to integer values
    train_labels = [LABEL_DICT[f_name.split('/')[0]] for f_name in train_filenames]

    # DataGenerator.__init__ requires num_classes -- this used to omit it
    # and crash with "missing 1 required positional argument: 'num_classes'"
    # the moment this file was run directly.
    data_gen = DataGenerator(train_filenames, train_labels, batch_size=32, num_classes=len(LABEL_DICT))
    batch_x, batch_y = data_gen[0]
    # numpy array of three 250 x 250 channels
    print(batch_x[0])
    # result should be one hot encoded
    print(batch_y[0])
    print(train_filenames[0])
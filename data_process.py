# Script to pre process bone marrow cell dataset
# name: data_process.py
# author: mbwhiteh@sfu.ca
# date: 2022-02-21

import os
import shutil
import random
import math

# set pseudo-random generator
random.seed(5)

DATASET_NAME = "BMC-Dataset"
# DANGER will modify dataset directory
def RenameNSort(dataset_pathname):
    # Class Labels in Directory
    classes_dirname = os.listdir(dataset_pathname)
    for class_dirname in classes_dirname:
        class_root = os.path.join(dataset_pathname, class_dirname)
        if(os.path.isdir(class_root)):
            # iterate through class root directory
            for child_class_dirname in os.listdir(class_root):
                child_class_path = os.path.join(class_root, child_class_dirname)
                # if directory open directory and move files on level up
                if(os.path.isdir(child_class_path)):
                    for file_inst in os.listdir(child_class_path):
                        f_src_path = os.path.join(child_class_path, file_inst)
                        # move the file one level up
                        f_des_path = os.path.join(class_root, file_inst)
                        try:
                            # if file does not already exist copy
                            if(not os.path.exists(f_des_path)):
                                shutil.copyfile(f_src_path, f_des_path)
                        except(shutil.SameFileError):
                            if(file_inst == '.DS_Store'):
                                os.remove(f_src_path)
                        # remove directory as all files have been copied
                        # os.removedirs(child_class_path)
                        print(child_class_path)
                        print("{0} -> {1}".format(f_src_path, f_des_path))
                    # all data files moved, ok to remove sub directory
                    shutil.rmtree(child_class_path)

def DatasetStats(dataset_path, output_filename):
    # obtain class labels and file counts for each label
    classnames = os.listdir(dataset_path)
    data_dict = dict()
    # total file count
    total_count = 0
    for classname in classnames:
        class_dir_path = os.path.join(dataset_path, classname)
        try:
            data_dict[classname] = len(os.listdir(class_dir_path))
            total_count += len(os.listdir(class_dir_path))
        except(NotADirectoryError):
            print("NotADirectory Exception")
    # write the dataset stats to a csv file
    with open(os.path.join(os.path.abspath('./'), output_filename), "w+") as fd:
        fd.write("ClassLabel, ImageCount, PercentageOfDataset\n")
        for cname in data_dict:
            fd.write("{0}, {1}, {2:.4f}\n".format(cname, data_dict[cname], (data_dict[cname]/total_count)*100))

# Returns two dictionaries (train_set, test_set) where the key (classname) in both dictionaries maps
# to a list of filenames that correspond to images.
"""
Parameters: 
    dataset_path (string) - absolute path to dataset
    train_test_split - fractional value between 0 to 1 to determine the fraction
    of data used for training.
Returns:
    (train_set, test_set) - a tuple of lists storing testing and training dataset filenames.
"""
def GenerateTrainTestFilenames(dataset_path, train_test_split):
    classnames = ["BLA", "LYT", "NGB", "NGS"]
    train_set = list()
    test_set = list()
    for classname in classnames:
        class_dir_path = os.path.join(dataset_path, classname)
        filenames = os.listdir(class_dir_path)
        train_data_filenames = random.sample(filenames, math.floor(len(filenames)*train_test_split))
        test_data_filenames = list()
        for filename in filenames:
            if(filename not in train_data_filenames):
                test_set.append(os.path.join(classname, filename))
            else:
                train_set.append(os.path.join(classname, filename))
    return train_set, test_set

if __name__ == "__main__":
    # abs path to dataset
    datapath = os.path.join(os.path.abspath("./"), DATASET_NAME)
    # RenameNSort(datapath)
    # DatasetStats(datapath, 'dataset_stats.csv')
    train_filenames, test_filenames = GenerateTrainTestFilenames(datapath, 0.9)
    # write both training and testing filenames to text file
    with open('train_filenames.txt', 'w+') as train_fd:
        train_fd.writelines([ f_name + '\n' for f_name in train_filenames ])
    with open('test_filenames.txt', 'w+') as test_fd:
        test_fd.writelines([ f_name + '\n' for f_name in test_filenames ])
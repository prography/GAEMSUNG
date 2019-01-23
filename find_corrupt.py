import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2


#os.remove(path) #Delete file
#os.removedirs(path) #Delete empty folder


def find_corrupt(folder_path):
    data_dir = folder_path
    flds = os.listdir(data_dir)

    for fld in flds:
        sub_flds = os.listdir(data_dir + '/' + fld)
        print(len(sub_flds))

        if len(sub_flds) == 0:
            print(fld)
     
        for i in sub_flds:
            i_path = data_dir + '/' + fld + '/' + i
            img = imread(i_path, 0)
            print(img.shape[-1])
            if img.shape[-1] != 3:
                print(i_path)
                os.remove(i_path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="____")
    PARSER.add_argument('-f', '--folder_path')
    ARGS = PARSER.parse_args()
    find_corrupt(str(ARGS.folder_path))

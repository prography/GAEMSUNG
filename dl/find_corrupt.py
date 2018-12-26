import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

#os.remove(path) #Delete file
#os.removedirs(path) #Delete empty folder


def find_corrupt(folder_path):
    data_dir = folder_path
    flds = os.listdir(data_dir)

    for fld in flds:
        sub_flds = os.listdir(data_dir + '/' + fld)
        if len(sub_flds) == 0:
            print(fld)
        try:
            for i in sub_flds:
                i_path = data_dir + '/' + fld + '/' + i
                img = imread(i_path)
                #print(np.shape(img))
        except:
            print(i_path)
            os.remove(i_path)  #Delete folders


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="____")
    PARSER.add_argument('-f', '--folder_path')
    ARGS = PARSER.parse_args()
    find_corrupt(str(ARGS.folder_path))
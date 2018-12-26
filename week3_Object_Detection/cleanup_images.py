"""
cleanup image file based on list file
"""
import cv2
import os, shutil

# read from list file and move images
def move_image_file(listfile, origin_root):
    with open(listfile, 'r') as f:
        listfile = f.readlines()

    for path in listfile:
        path = path.replace('\n', '')
        filename = os.path.split(path)[-1]
        cls_, picname_ = filename.split('_')[:-2], filename.split('_')[-2:]

        cls, picname = '', ''

        for token in cls_:
            cls += token
            cls += '_'

        for token in picname_:
            picname += token
            picname += '_'

        cls, picname = cls[:-1], picname[:-1] # remove last underbar
        origin_path = os.path.join(origin_root, cls, picname)
        shutil.copy(origin_path, path) # copy images
        print('Copy completed!')

if __name__ == '__main__':
    move_image_file('trainList.txt', 'D:\Deep_learning\Data\\the-simpsons-characters-dataset\simpsons_dataset')
    move_image_file('evalList.txt', 'D:\Deep_learning\Data\\the-simpsons-characters-dataset\simpsons_dataset')

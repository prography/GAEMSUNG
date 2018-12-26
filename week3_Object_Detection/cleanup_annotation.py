"""
clean up annotation file, list file, class name file
"""
from random import shuffle
import os

def shuffle_and_split(file, ratio):
    # 0. read from original annotation file
    with open(file, 'r') as f:
        anno = f.readlines()

    # 1. shuffle rows
    shuffle(anno)

    # 2. split train/eval annotations
    train_len = int(len(anno)*ratio)
    train_anno = anno[:train_len]
    eval_anno = anno[train_len:]

    # 3. write annotation files
    with open('trainAnno.txt', 'w') as train:
        train.writelines(train_anno)

    with open('evalAnno.txt', 'w') as eval:
        eval.writelines(eval_anno)

    print("Make annotation file completed!")

def make_annotation_files(file, img_root, cls2idx, imglist_outf, label_outf):
    # 0. read from annotation file
    with open(file, 'r') as f:
        anno = f.readlines()

    # 1. build class2idx vocab
    if cls2idx is None:
        cls2idx = {}

    for line in anno:
        line = line.replace('\n', '')
        tokens = line.split(',')
        class_name = tokens[5]

        if class_name not in cls2idx:
            cls2idx[class_name] = len(cls2idx)

    # 2. make class name file
    if 'eval' in file:
        for cls in cls2idx.keys():
            with open('simpson.names', 'a') as f:
                f.writelines(cls + '\n')

        print("Make name file!")

    # 3. make filename list file
    for line in anno:
        line = line.replace('\n', '')
        tokens = line.split(',')
        filename = tokens[0]
        class_name = tokens[5]
        output_name = class_name + '_' + filename.split('/')[-1]

        with open(imglist_outf, 'a') as f:
            f.write(os.path.join(img_root, output_name) + '\n')

    print("Make list file!")

    # 3. make annotation files (labels directory)
    for line in anno:
        # 1. remove new line
        line = line.replace('\n', '')

        # 2. split by comma
        tokens = line.split(',')
        filename = tokens[0]
        x1, y1, x2, y2 = int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4])
        class_name = tokens[5]

        output_name = class_name + '_' + filename.split('/')[-1].replace('.jpg', '.txt')

        width = x2 - x1
        height = y2 - y1
        newline = str(cls2idx[class_name]) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(width) + ' ' + str(height)

        # 3. save files
        with open(os.path.join(label_outf, output_name), 'w') as f:
            f.write(newline)

    print("Make all annotation files!")

    return cls2idx

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    train_imgs_path = 'images/train'
    eval_imgs_path = 'images/eval'
    train_labels_path = 'labels/train'
    eval_labels_path = 'labels/eval'

    # make directories
    make_directory(train_imgs_path)
    make_directory(eval_imgs_path)
    make_directory(train_labels_path)
    make_directory(eval_labels_path)

    shuffle_and_split("D:\Deep_learning\Data\\the-simpsons-characters-dataset\\annotation.txt", 0.7)

    cls2idx = make_annotation_files('trainAnno.txt', train_imgs_path, None, 'trainList.txt', train_labels_path)
    _ = make_annotation_files('evalAnno.txt', eval_imgs_path, cls2idx, 'evalList.txt', eval_labels_path)
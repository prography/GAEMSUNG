from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from mobilenetv2 import MobileNetV2

import csv

def main(args):    
    # set cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    print('[*] Preparing Data...')
    
    # make training data transforms.
    transform_train = transforms.Compose([transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # make test data transforms.
    transforms_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # load training data.
    trainset = torchvision.datasets.ImageFolder(root='datasets/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.ImageFolder(root='datasets/train', transform=transforms_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # Model
    print('[*] Building model...')
    net = MobileNetV2(n_class=1000)
    state_dict = torch.load('models/mobilenet_v2.pth.tar')
    net.load_state_dict(state_dict)

    if device  == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    outputs = net(images)
    print('output shape: ', outputs.data.shape)

    with open('cafeimage_embeddings.csv','wb') as f:
        for key in outputs.keys():
            f.write("%s,"%(outputs[key]))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=10e-4, type=float, help='learning rate')
    parser.add_argument('--image_size', default=228, type=int, help='input image size')

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
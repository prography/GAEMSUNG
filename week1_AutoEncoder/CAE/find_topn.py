"""
input: one image
output: 5 images
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import os
import numpy as np
import PIL.Image as Image
import cv2

from dataloader import get_loader
from model import ConvolutionalAE
from config import get_config

class Finder(object):
    def __init__(self, config):
        self.num_find = config.num_find
        self.img_path = config.img_path
        # self.candidate_path = config.candidate_path
        self.model_path = config.model_path

        # self.candidate_list = os.listdir(self.candidate_path)

        # if self.num_find > len(self.candidate_list):
        #     print("[*] Too large num_find!")
        #     exit()

        self.image_size = config.image_size
        self.in_channel = config.in_channel
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.dataset = CIFAR10(config.dataroot, train=False, transform=self.transform, download=True)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

        self.load_net()

    def load_net(self):
        # define network
        self.net = ConvolutionalAE(self.in_channel, self.hidden_dim, self.output_dim)

        # load pretrained state dict
        if self.model_path == None:
            print("[*] ERROR! Please enter weight path!")
        else:
            self.net.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
            print("[*] Load state dict from {}".format(self.model_path))

        self.net.to(self.device)
        self.net.eval()

    def find_topn(self):
        input_image = self.transform(Image.open(self.img_path)).unsqueeze(0)
        embedded = self.net.encoder(input_image)

        dist_list = []
        criterion = nn.MSELoss().to(self.device)

        # for path in self.candidate_list:
        #     candid_image = self.transform(Image.open(path))
        #     embedded_candid = self.net.encoder(candid_image)
        #     print(embedded_candid)
        #     dist = criterion(embedded_candid, embedded)
        #     dist_list.append(dist.item())

        dataiter = iter(self.dataloader)

        for candid_image, _ in dataiter:
            candid_image = candid_image.to(self.device)
            embedded_candid = self.net.encoder(candid_image)
            # print(embedded_candid)
            dist = criterion(embedded_candid, embedded)
            # print(dist.item())
            dist_list.append(dist.item())

        dist_argsorted = np.argsort(dist_list)

        for i in range(self.num_find):
            print("arg:", dist_argsorted[i])

            img, _ = self.dataset[dist_argsorted[i]]
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))

            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])

            cv2.imshow('Similar image', img)
            cv2.waitKey(0)

if __name__ == '__main__':
    config = get_config()
    finder = Finder(config)
    finder.find_topn()







import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import os
import numpy as np
import PIL.Image as Image
import cv2

from dataloader import get_loader
from model import ConvolutionalAE
from config import get_config
from torchvision.utils import save_image

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

        self.dataset = MNIST(config.dataroot, train=False, transform=self.transform, download=True)
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

        eu_dist_list = []
        ae_dist_list = []
        criterion = nn.MSELoss().to(self.device)

        # for path in self.candidate_list:
        #     candid_image = self.transform(Image.open(path))
        #     embedded_candid = self.net.encoder(candid_image)
        #     print(embedded_candid)
        #     dist = criterion(embedded_candid, embedded)
        #     dist_list.append(dist.item())

        dataiter = iter(self.dataloader)
        sample = next(dataiter)[0]
        embedded = self.net.encoder(sample) # 1, 20, 2, 2
        save_image(sample, "assets/sample.jpg", normalize=True)

        ### compare with euclidian distance ###
        for candid_image, _ in dataiter:
            # print(candid_image.shape)
            # print(sample.shape)
            eu_dist = torch.dist(candid_image, sample)
            eu_dist_list.append(eu_dist.item())

        dataiter = iter(self.dataloader)

        ### compare with auto encoder ###
        for candid_image, _ in dataiter:
            candid_image = candid_image.to(self.device)
            embedded_candid = self.net.encoder(candid_image) # 1, 20, 2, 2
            dist = criterion(embedded_candid, embedded)

            # print(dist.item())
            ae_dist_list.append(dist.item())

        eu_dist_argsorted = np.argsort(eu_dist_list)
        ae_dist_argsorted = np.argsort(ae_dist_list)

        print(eu_dist_argsorted)
        print(ae_dist_argsorted)

        for i in range(self.num_find):
            print("arg:", eu_dist_argsorted[i])

            img, _ = self.dataset[eu_dist_argsorted[i]]
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))

            # r, g, b = cv2.split(img)
            # img = cv2.merge([b, g, r])

            cv2.imshow('Similar image computed by EU dist', img)
            cv2.waitKey(0)
            # cv2.imwrite("assets/EU_similar_%d.jpg" % i, img)


        for i in range(self.num_find):
            print("arg:", ae_dist_argsorted[i])

            img, _ = self.dataset[ae_dist_argsorted[i]]
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))

            # r, g, b = cv2.split(img)
            # img = cv2.merge([b, g, r])

            cv2.imshow('Similar image computed by AE dist', img)
            cv2.waitKey(0)
            # cv2.imwrite("assets/AE_similar_%d.jpg" % i, img)


if __name__ == '__main__':
    config = get_config()
    finder = Finder(config)
    finder.find_topn()
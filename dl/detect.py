"""
input: a path to input image
outputs: True(chair detected) or False(chair not detected)
"""
from __future__ import division
from models import *

import os

import torch
import torchvision.transforms as T
from utils.utils import *
from PIL import Image

class yoloDetector():
    def __init__(self):
        # hyper parameters
        self.config_path = 'config/yolov3.cfg'
        self.weights_path = 'weights/yolov3.weights'
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.image_size = 416
        self.use_cuda = False
        self.model = Darknet(self.config_path, img_size=self.image_size)
        
        if self.use_cuda:
            self.model.cuda()

    def load_weights(self):
        self.model.load_weights(self.weights_path)
    
    def detect(self, input_image_path):
        # prepare input image tensor
        image = Image.open(input_image_path)
        transforms = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor()
        ])
        image_tensor = transforms(image).unsqueeze(0)

        imgs = []           # Stores image paths
        img_detections = [] # Stores detections for each image index

        # Get detections
        with torch.no_grad():
            detections = self.model(image_tensor)
            detections = non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)

        # Save image and detections
        imgs.append(input_image_path)
        img_detections.extend(detections)

        # Iterate through images and save plot of detections
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            # Draw bounding boxes and labels of detections
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                if 56. in unique_labels:
                    return True
                else:
                    return False

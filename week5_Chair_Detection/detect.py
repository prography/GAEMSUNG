"""
input: a path to input image
outputs: True(chair detected) or False(chair not detected)
"""

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime

import torch
import torchvision.transforms as T
from PIL import Image

"""
using sys.argv 
python detect.py <input_image_path> --> output log message: True/False/None
"""
def main():
    # hyper parameters
    config_path = 'config/yolov3.cfg'
    weights_path = 'weights/yolov3.weights'
    conf_thres = 0.8
    nms_thres = 0.4
    image_size = 416
    use_cuda = False

    # Set up model
    model = Darknet(config_path, img_size=image_size)
    model.load_weights(weights_path)
    print("Load trained weight file completed!!!")

    input_image_path = sys.argv[1]

    if use_cuda:
        model.cuda()

    model.eval() # Set in evaluation mode

    # prepare input image tensor
    image = Image.open(input_image_path)
    transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    image_tensor = transforms(image).unsqueeze(0)

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index

    print ("[*] Performing object detection:")
    prev_time = time.time()

    # Get detections
    with torch.no_grad():
        detections = model(image_tensor)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    print ('\t+ Inference Time: %s' % (inference_time))

    # Save image and detections
    imgs.append(input_image_path)
    img_detections.extend(detections)

    print ('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            if 56. in unique_labels:
                return True
            else:
                return False

if __name__ == '__main__':
    detect_result = main()
    print("Are chairs in image? ", detect_result)
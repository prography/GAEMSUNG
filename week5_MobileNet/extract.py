"""
Pretrained model에서 feature extraction network 부분만 뽑아서
feature map 뽑아내는 부분
"""

import torch
from models.imagenet.mobilenetv2 import mobilenetv2
import torchvision.transforms as T
from PIL import Image
import sys

"""
일단은 parameter로 이미지 한 장의 경로를 입력하게 구현
--> Directory 단위로 돌려야 하면 input 가져오는 부분만 수정하면 됨.
--> 전체적인 흐름 알려주기 위해 짠 코드니까 언제든지 수정하고 확인 받기!
"""
def main(image_path):
    # hyper parameters
    weight_path = 'weight/mobilenetv2.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load trained model
    net = mobilenetv2()
    net.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
    print("[*]Load state dict completed!!!")

    # switch to eval mode and get only feature extraction network
    net.eval()
    feature_net = net.features.to(device)
    print(feature_net)

    # load image tensor
    image = Image.open(image_path)
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalize value
    ])
    image_tensor = transforms(image).unsqueeze(0)

    # forwarding to feature extraction network
    image_tensor = image_tensor.to(device)
    features = feature_net(image_tensor)

    print(features)
    print(features.shape)
    return features

if __name__ == '__main__':
    """
    Usage: python extract.py <input_image_path>
    EX> python extract.py inputs/caffe001.jpg
    """
    features = main(sys.argv[1])





from torchvision.models import resnet18
import torchvision.transforms as T
from PIL import Image

import torch.nn as nn

"""
Feature extraction network만 떼서 가져오기
"""
net = resnet18(pretrained=True)
modules = list(net.children())[:-1] # feature extraction network
features = nn.Sequential(*modules)
for p in features.parameters():
    p.requires_grad = False

frame = Image.open('C:\\Users\young\Pictures\Camera Roll\WIN_20180817_16_51_53_Pro.jpg')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5))
]
)
frame_t = transform(frame).unsqueeze(0)
print(frame_t.shape)

output = features(frame_t)
print(output.shape)


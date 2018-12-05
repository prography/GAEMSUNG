import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_channels, ndf, num_classes):
        super(Classifier, self).__init__()
        self.num_channels = num_channels
        self.ndf = ndf
        self.num_classes = num_classes

        # feature extraction network
        self.feature = nn.Sequential(
            nn.Conv2d(num_channels, ndf, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ndf, ndf*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # state output: batch, 32, 7, 7

        # classifier network
        self.clf = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        feature_out = self.feature(x) # forwarding to feature extraction network
        feature_out = feature_out.view(feature_out.size(0), -1) # flatten
        clf_out = self.clf(feature_out) # forwarding to fully-connected network
        return clf_out
"""
Modify first FC layer's input size and last FC layer's output size from (image size)**2 to (image size)**2*(in channel)
because CIFAR10 image dataset has 3 channel images.
"""

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channel, image_size, hidden_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # define encoder network
        self.encoder = nn.Sequential(
            nn.Linear(in_channel*(image_size**2), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )

        # define decoder network
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channel*(image_size**2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out
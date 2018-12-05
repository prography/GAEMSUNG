import torch.nn as nn

class ConvolutionalAE(nn.Module):
    def __init__(self, in_channel, hidden_dim, output_dim):
        super(ConvolutionalAE, self).__init__()
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim, kernel_size=3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        # decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(output_dim, hidden_dim, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(output_dim, in_channel, kernel_size=2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out
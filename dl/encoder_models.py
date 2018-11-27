import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # define encoder network
        self.encoder = nn.Sequential(
            nn.Linear(image_size**2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )

        # define decoder network
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, image_size**2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out

class VAE(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim):
        super(VAE, self).__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # encoder network
        self.fc1 = nn.Linear(image_size ** 2, hidden_dim)
        self.fc2_1 = nn.Linear(hidden_dim, output_dim)
        self.fc2_2 = nn.Linear(hidden_dim, output_dim)

        # decoder network
        self.fc3 = nn.Linear(output_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_size ** 2)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2_1(h1), self.fc2_2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h4 = self.sigmoid(self.fc4(h3))
        return h4

    def forward(self, x):
        mu, logvar = self.encode(x) # forwarding to encoder
        z = self.reparameterize(mu, logvar) # reparameterize
        out = self.decode(z)  # forwarding to decoder
        return out, mu, logvar
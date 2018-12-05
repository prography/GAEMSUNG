import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_channel, image_size, hidden_dim, output_dim):
        super(VAE, self).__init__()
        self.in_channel = in_channel
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # encoder network
        self.fc1 = nn.Linear(in_channel*(image_size ** 2), hidden_dim)
        self.fc2_1 = nn.Linear(hidden_dim, output_dim)
        self.fc2_2 = nn.Linear(hidden_dim, output_dim)

        # decoder network
        self.fc3 = nn.Linear(output_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, in_channel*(image_size ** 2))

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
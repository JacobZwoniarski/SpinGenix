# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, param_dim, grid_size, latent_dim=128):
        super(ConditionalVAE, self).__init__()
        self.param_dim = param_dim
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (grid_size // 4) ** 2, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 + param_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 + param_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + param_dim, 64 * (grid_size // 4) ** 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x, params):
        h = self.encoder(x)
        h = torch.cat([h, params], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, params):
        z = torch.cat([z, params], dim=1)
        h = self.decoder_input(z)
        h = h.view(-1, 64, self.grid_size // 4, self.grid_size // 4)
        return self.decoder(h)

    def forward(self, x, params):
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, params), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=0.01):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD, MSE, KLD

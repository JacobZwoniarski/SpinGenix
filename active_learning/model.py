import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Basic blocks: ConvBlock, Down, Up
# -----------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # FiLM parameters generated from cond vector (Tx, Tz)
        self.scale = nn.Linear(cond_dim, out_channels)
        self.shift = nn.Linear(cond_dim, out_channels)

    def forward(self, x, cond):
        # FiLM modulation
        gamma = self.scale(cond).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta = self.shift(cond).unsqueeze(-1).unsqueeze(-1)   # (B,C,1,1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x + gamma + beta)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x + gamma + beta)

        return x


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)


# -----------------------------------------------------------
# UNet–CVAE
# -----------------------------------------------------------

class UNetCVAE(nn.Module):
    def __init__(self, spatial_size=200, latent_dim=64, cond_dim=2):
        super().__init__()

        self.spatial_size = spatial_size
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Encoder blocks
        self.enc1 = ConvBlock(3, 32, cond_dim)
        self.down1 = Down(32, 64)

        self.enc2 = ConvBlock(64, 128, cond_dim)
        self.down2 = Down(128, 256)

        self.enc3 = ConvBlock(256, 256, cond_dim)
        self.down3 = Down(256, 256)

        self.enc4 = ConvBlock(256, 256, cond_dim)

        # -------------------------------------------------------
        # Dynamically infer the flatten dimension using dummy pass
        # -------------------------------------------------------
        with torch.no_grad():
            dummy_x = torch.zeros(1, 3, spatial_size, spatial_size)
            dummy_c = torch.zeros(1, cond_dim)
            out, _ = self.encode_conv_only(dummy_x, dummy_c)
            self.enc_out_shape = out.shape[1:]   # (C,H,W)
            self.flat_dim = out.numel()          # total size

        # Latent projection
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder initial projection
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # Decoder blocks
        C, H, W = self.enc_out_shape

        self.up1 = Up(256, 256)
        self.dec1 = ConvBlock(512, 256, cond_dim)

        self.up2 = Up(256, 128)
        self.dec2 = ConvBlock(256, 128, cond_dim)

        self.up3 = Up(128, 64)
        self.dec3 = ConvBlock(96, 64, cond_dim)

        self.dec4 = ConvBlock(64, 32, cond_dim)
        self.final = nn.Conv2d(32, 3, 1)


    # -----------------------------------------------------------
    # Helper: encoder WITHOUT flatten or linear layers
    # -----------------------------------------------------------
    def encode_conv_only(self, x, cond):
        skips = []

        x = self.enc1(x, cond)
        skips.append(x)
        x = self.down1(x)

        x = self.enc2(x, cond)
        skips.append(x)
        x = self.down2(x)

        x = self.enc3(x, cond)
        skips.append(x)
        x = self.down3(x)

        x = self.enc4(x, cond)
        skips.append(x)

        return x, skips


    # -----------------------------------------------------------
    # Full encoder
    # -----------------------------------------------------------
    def encode(self, x, cond):
        out, skips = self.encode_conv_only(x, cond)
        flat = out.view(out.size(0), -1)

        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)

        return mu, logvar, skips


    # -----------------------------------------------------------
    # Reparameterization trick
    # -----------------------------------------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    # -----------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------
    def decode(self, z, skips, cond):

        out = self.fc_decode(z)
        out = out.view(-1, *self.enc_out_shape)

        # Decoder levels
        out = self.up1(out)
        out = torch.cat([out, skips[-2]], dim=1)
        out = self.dec1(out, cond)

        out = self.up2(out)
        out = torch.cat([out, skips[-3]], dim=1)
        out = self.dec2(out, cond)

        out = self.up3(out)
        out = torch.cat([out, skips[-4]], dim=1)
        out = self.dec3(out, cond)

        out = self.dec4(out, cond)
        out = self.final(out)
        out = torch.tanh(out)

        return out


    # -----------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------
    def forward(self, x, cond):
        mu, logvar, skips = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, skips, cond)
        return out, mu, logvar
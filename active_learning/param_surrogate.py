import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x * (1.0 + gamma) + beta)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x + residual)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.block = ConditionalResBlock(out_channels, cond_dim)

    def forward(self, x, cond):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.block(x, cond)


class ConditionalResNetDecoder(nn.Module):
    """
    Deterministic surrogate baseline: normalized parameters -> field.

    The model predicts the canonical SpinGenix field representation
    (B, 3, 200, 200). Physical dimensions remain in metadata/registry.
    """

    def __init__(
        self,
        spatial_size=200,
        cond_dim=2,
        hidden_dim=128,
        base_channels=32,
        start_size=None,
    ):
        super().__init__()
        if start_size is None:
            if spatial_size % 8 != 0:
                raise ValueError("spatial_size must be divisible by 8 when start_size is omitted.")
            start_size = spatial_size // 8

        self.spatial_size = int(spatial_size)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.base_channels = int(base_channels)
        self.start_size = int(start_size)
        self.start_channels = self.base_channels * 8

        self.cond_encoder = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.seed = nn.Linear(
            self.hidden_dim,
            self.start_channels * self.start_size * self.start_size,
        )

        self.start_block = ConditionalResBlock(self.start_channels, self.hidden_dim)
        self.up1 = UpsampleBlock(self.start_channels, self.base_channels * 4, self.hidden_dim)
        self.up2 = UpsampleBlock(self.base_channels * 4, self.base_channels * 2, self.hidden_dim)
        self.up3 = UpsampleBlock(self.base_channels * 2, self.base_channels, self.hidden_dim)
        self.out = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.base_channels, 3, kernel_size=1),
            nn.Tanh(),
        )

    def config(self):
        return {
            "spatial_size": self.spatial_size,
            "cond_dim": self.cond_dim,
            "hidden_dim": self.hidden_dim,
            "base_channels": self.base_channels,
            "start_size": self.start_size,
        }

    def forward(self, params):
        cond = self.cond_encoder(params)
        x = self.seed(cond)
        x = x.view(
            params.shape[0],
            self.start_channels,
            self.start_size,
            self.start_size,
        )
        x = self.start_block(x, cond)
        x = self.up1(x, cond)
        x = self.up2(x, cond)
        x = self.up3(x, cond)
        if x.shape[-1] != self.spatial_size or x.shape[-2] != self.spatial_size:
            x = F.interpolate(
                x,
                size=(self.spatial_size, self.spatial_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.out(x)

    @torch.no_grad()
    def sample(self, params):
        return self.forward(params)

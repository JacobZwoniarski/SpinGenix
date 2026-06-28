import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


# ------------------------------------------------------------
# Weighted Reconstruction Loss (Option B)
# ------------------------------------------------------------
class WeightedMSELoss(nn.Module):
    def __init__(self, wx=1.0, wy=1.0, wz=2.0):
        super().__init__()
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, recon, target):
        # recon/target: (B, 3, H, W)
        loss_x = self.mse(recon[:, 0], target[:, 0])
        loss_y = self.mse(recon[:, 1], target[:, 1])
        loss_z = self.mse(recon[:, 2], target[:, 2])

        return (
            self.wx * loss_x +
            self.wy * loss_y +
            self.wz * loss_z
        )


# ------------------------------------------------------------
# Kullback–Leibler Divergence (VAE)
# ------------------------------------------------------------
def kl_divergence(mu, logvar):
    # Standard KL term with mean reduction
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ------------------------------------------------------------
# CVAE Trainer (fixed for CUDA multiprocessing)
# ------------------------------------------------------------
def train_cvae(
    model,
    dataset,
    epochs=20,
    batch_size=8,
    lr=1e-4,
    beta=0.001,               # KL weight
    wx=1.0,
    wy=1.0,
    wz=2.0,
    device="cuda",
    checkpoint_callback=None,
):
    # Move model to GPU
    model = model.to(device)
    model.train()

    # IMPORTANT FIX:
    # CUDA CANNOT BE INITIALIZED IN DATALOADER WORKERS!
    # Use num_workers=0 during tests or HPC runs.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0   # <-- FIXED (0 avoids CUDA fork issues)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rec_loss_fn = WeightedMSELoss(wx, wy, wz)

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for fields, params in pbar:
            # IMPORTANT FIX:
            # Move tensors to CUDA here, not in dataset.
            fields = fields.to(device)
            params = params.to(device)

            optimizer.zero_grad()

            recon, mu, logvar = model(fields, params)

            rec_loss = rec_loss_fn(recon, fields)
            kl = kl_divergence(mu, logvar)
            loss = rec_loss + beta * kl

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (len(pbar) + 1)})

        epoch_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}] loss = {epoch_loss:.6f}", flush=True)
        model._last_epoch_loss = epoch_loss
        if checkpoint_callback is not None:
            checkpoint_callback(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=epoch_loss,
            )

    model._last_optimizer_state_dict = optimizer.state_dict()
    return model

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


class MaskedWeightedMSELoss(nn.Module):
    def __init__(self, wx=1.0, wy=1.0, wz=2.0, mask_threshold=1e-4, background_weight=0.05):
        super().__init__()
        self.channel_weights = torch.tensor([wx, wy, wz], dtype=torch.float32).view(1, 3, 1, 1)
        self.mask_threshold = float(mask_threshold)
        self.background_weight = float(background_weight)

    def forward(self, recon, target):
        weights = self.channel_weights.to(device=recon.device, dtype=recon.dtype)
        squared = (recon - target).pow(2) * weights
        target_norm = torch.linalg.norm(target, dim=1, keepdim=True)
        material_mask = target_norm > self.mask_threshold
        spatial_weight = torch.where(
            material_mask,
            torch.ones_like(target_norm),
            torch.full_like(target_norm, self.background_weight),
        )
        return torch.sum(squared * spatial_weight) / torch.clamp(torch.sum(spatial_weight) * recon.shape[1], min=1.0)


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


# ------------------------------------------------------------
# Param-to-field surrogate trainer
# ------------------------------------------------------------
def surrogate_unit_norm_penalty(prediction, target=None, mask_threshold=1e-4):
    norm = torch.linalg.norm(prediction, dim=1)
    if target is not None:
        target_norm = torch.linalg.norm(target, dim=1)
        material_mask = target_norm > mask_threshold
        if torch.any(material_mask):
            return torch.mean(torch.abs(norm[material_mask] - 1.0))
    return torch.mean(torch.abs(norm - 1.0))


def train_param_surrogate(
    model,
    dataset,
    epochs=20,
    batch_size=8,
    lr=1e-4,
    norm_weight=0.01,
    mask_threshold=1e-4,
    device="cuda",
    checkpoint_callback=None,
):
    """Train a deterministic params -> canonical field surrogate."""

    model = model.to(device)
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rec_loss_fn = MaskedWeightedMSELoss(mask_threshold=mask_threshold)

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"ParamSurrogate Epoch {epoch+1}/{epochs}")

        for fields, params in pbar:
            fields = fields.to(device)
            params = params.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(params)
            rec_loss = rec_loss_fn(pred, fields)
            norm_loss = surrogate_unit_norm_penalty(
                pred,
                fields,
                mask_threshold=mask_threshold,
            )
            loss = rec_loss + norm_weight * norm_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / max(len(pbar), 1)})

        epoch_loss = total_loss / max(len(loader), 1)
        print(f"[ParamSurrogate Epoch {epoch+1}] loss = {epoch_loss:.6f}", flush=True)
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class WeightedMSELoss(nn.Module):
    def __init__(self, wx=1.0, wy=1.0, wz=2.0):
        super().__init__()
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, recon, target):
        loss_x = self.mse(recon[:, 0], target[:, 0])
        loss_y = self.mse(recon[:, 1], target[:, 1])
        loss_z = self.mse(recon[:, 2], target[:, 2])
        return self.wx * loss_x + self.wy * loss_y + self.wz * loss_z


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_cvae(
    model,
    dataset,
    epochs=20,
    batch_size=8,
    lr=1e-4,
    beta=0.001,
    wx=1.0,
    wy=1.0,
    wz=2.0,
    device="cuda"
):
    model = model.to(device)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # ważne na HPC + CUDA
        pin_memory=device.startswith("cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rec_loss_fn = WeightedMSELoss(wx, wy, wz)

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            # batch może być (fields, params) albo (fields, params, meta)
            if len(batch) == 2:
                fields, params = batch
            else:
                fields, params, _meta = batch

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
            pbar.set_postfix({"loss": total_loss / (epoch + 1)})

        print(f"[Epoch {epoch+1}] loss = {total_loss / len(loader):.6f}")

    return model
import torch
import numpy as np
from tqdm import tqdm


def _norm(name: str, val: float, param_ranges: dict | None):
    if param_ranges is None:
        return float(val)
    lo, hi = param_ranges[name]
    if hi == lo:
        return 0.0
    return float(2.0 * (val - lo) / (hi - lo) - 1.0)


def _enable_dropout_only(model: torch.nn.Module):
    """
    Dropout ON, ale bez przełączania BatchNorm w train (żeby nie psuć statystyk).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_forward(model, fields, cond, mc_samples=10):
    _enable_dropout_only(model)

    preds = []
    for _ in range(mc_samples):
        out, _, _ = model(fields, cond)
        preds.append(out)

    return torch.stack(preds, dim=0)  # (N,B,3,H,W) jeśli fields ma batch B


@torch.no_grad()
def compute_uncertainty_point(model, Tx, Tz, device="cuda", mc_samples=10, H=200, W=200, param_ranges=None):
    dummy = torch.zeros((1, 3, H, W), device=device)

    tx_n = _norm("Tx_val", float(Tx), param_ranges)
    tz_n = _norm("Tz_val", float(Tz), param_ranges)
    cond = torch.tensor([[tx_n, tz_n]], dtype=torch.float32, device=device)

    preds = mc_dropout_forward(model, dummy, cond, mc_samples=mc_samples)  # (N,1,3,H,W)
    preds = preds[:, 0]  # -> (N,3,H,W)

    var_map = torch.var(preds, dim=0)  # (3,H,W)
    aggregate = torch.mean(torch.norm(var_map, dim=0))  # scalar
    return aggregate.item()


@torch.no_grad()
def compute_uncertainty_map(model, Tx_grid, Tz_grid, mc_samples=10, device="cuda", param_ranges=None):
    model.to(device)

    H = 200
    W = 200
    U = np.zeros((len(Tx_grid), len(Tz_grid)), dtype=np.float32)

    for i, Tx in enumerate(tqdm(Tx_grid, desc="Uncertainty-Tx")):
        for j, Tz in enumerate(Tz_grid):
            U[i, j] = compute_uncertainty_point(
                model,
                float(Tx),
                float(Tz),
                device=device,
                mc_samples=mc_samples,
                H=H,
                W=W,
                param_ranges=param_ranges,
            )

    return U

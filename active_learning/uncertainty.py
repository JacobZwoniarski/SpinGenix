import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def mc_dropout_forward(model, fields, cond, mc_samples=10):
    """
    Runs the stochastic CVAE forward pass mc_samples times.
    Returns a stacked tensor of shape (mc_samples, C, H, W)
    """
    model.eval()

    preds = []
    for _ in range(mc_samples):
        out, _, _ = model(fields, cond)
        preds.append(out)

    return torch.stack(preds, dim=0)  # (N, 3, H, W)


@torch.no_grad()
def compute_uncertainty_point(
    model,
    Tx,
    Tz,
    device="cuda",
    mc_samples=10,
    H=200,
    W=200,
    param_normalizer=None,
):
    """
    Computes uncertainty for a SINGLE (Tx, Tz) parameter point.

    Returns:
      scalar uncertainty value
    """

    # Create dummy zero-field input (model ignores input content thanks to conditioning)
    dummy = torch.zeros((1, 3, H, W), device=device)

    cond_values = np.array([Tx, Tz], dtype=np.float64)
    if param_normalizer is not None:
        cond_values = param_normalizer.transform(cond_values)
    cond = torch.tensor([cond_values], dtype=torch.float32, device=device)

    preds = mc_dropout_forward(model, dummy, cond, mc_samples=mc_samples)

    # Pixel-wise variance over MC samples
    var_map = torch.var(preds, dim=0)  # (3, H, W)

    # Aggregate uncertainty:
    # L2 norm over channels → average across space
    aggregate = torch.mean(torch.norm(var_map, dim=0))

    return aggregate.item()


@torch.no_grad()
def compute_uncertainty_map(
    model,
    Tx_grid,
    Tz_grid,
    mc_samples=10,
    device="cuda",
    param_normalizer=None,
):
    """
    Computes uncertainty over a 2D grid of parameters.

    Tx_grid, Tz_grid are 1D arrays → we compute Nx×Ny map.

    Returns:
        U : numpy array of shape (len(Tx_grid), len(Tz_grid))
    """

    model.to(device)
    model.eval()

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
                param_normalizer=param_normalizer,
            )

    return U

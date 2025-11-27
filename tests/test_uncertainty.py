import torch
from active_learning.model import UNetCVAE
from active_learning.uncertainty import compute_uncertainty_point

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetCVAE(spatial_size=200).to(device)

    U = compute_uncertainty_point(
        model,
        Tx=50e-9,
        Tz=35e-9,
        device=device,
        mc_samples=10
    )

    print("Uncertainty:", U)
    print("SUCCESS: Uncertainty calculation works.")
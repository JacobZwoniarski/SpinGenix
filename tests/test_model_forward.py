import torch
from active_learning.model import UNetCVAE

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetCVAE(spatial_size=200).to(device)

    dummy_field = torch.zeros((1, 3, 200, 200), device=device)
    cond = torch.tensor([[50e-9, 30e-9]], device=device)

    out, mu, logvar = model(dummy_field, cond)

    print("Output shape:", out.shape)      # Expect (1,3,200,200)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("SUCCESS: Model forward works.")
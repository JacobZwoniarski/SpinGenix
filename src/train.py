# src/train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import ConditionalVAE, loss_function
from utils import RealMagnetizationDatasetNew, load_dataframe_and_fields

def evaluate(model, data_loader, device):
    model.eval()
    total_loss, total_mse, total_kld = 0.0, 0.0, 0.0
    count = 0
    with torch.no_grad():
        for p, mag in data_loader:
            p = p.to(device)
            mag = mag.to(device)
            recon, mu, logvar = model(mag, p)
            loss, mse, kld = loss_function(recon, mag, mu, logvar)
            total_loss += loss.item()
            total_mse  += mse.item()
            total_kld  += kld.item()
            count      += 1
    avg_loss = total_loss / count if count else float('inf')
    avg_mse  = total_mse  / count if count else float('inf')
    avg_kld  = total_kld  / count if count else float('inf')
    return avg_loss, avg_mse, avg_kld

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, required=True, help="ścieżka do pliku HDF5 z metadanymi")
    parser.add_argument("--fields", type=str, required=True, help="ścieżka do pliku NPZ z polami magnetyzacji")
    parser.add_argument("--beta", type=float, default=0.01, help="Waga KLD")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--grid_size", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="model_ckpt.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaling_factors = {
        "Aex": 1e11,
        "Msat": 1e-5,
        "Tx": 1e9,
        "Tz": 1e9
    }
    param_dim = 4

    df = load_dataframe_and_fields(args.meta, args.fields)
    df_train, df_test = train_test_split(df, test_size=0.05, random_state=42)
    train_dataset = RealMagnetizationDatasetNew(df_train, scaling_factors)
    test_dataset = RealMagnetizationDatasetNew(df_test, scaling_factors)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ConditionalVAE(param_dim, args.grid_size, args.latent_dim).to(device)
    if torch.cuda.device_count() > 1:
        print("Używam DataParallel")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss, running_mse, running_kld = 0.0, 0.0, 0.0
        batches = 0
        for p, mag in train_loader:
            p = p.to(device)
            mag = mag.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(mag, p)
            loss, mse, kld = loss_function(recon, mag, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_mse  += mse.item()
            running_kld  += kld.item()
            batches      += 1
        train_loss = running_loss / batches if batches else float('inf')
        test_loss, test_mse, test_kld = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} (MSE: {test_mse:.4f}, KLD: {test_kld:.4f})")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model zapisany do {args.save_path}")

if __name__ == "__main__":
    main()

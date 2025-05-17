# src/train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from model import ConditionalVAE, loss_function
from utils import RealMagnetizationDatasetNew, load_dataframe_and_fields, convert_to_rgb

from torch.utils.tensorboard import SummaryWriter

def evaluate(model, data_loader, device, beta=0.01):
    model.eval()
    total_loss, total_mse, total_kld = 0.0, 0.0, 0.0
    count = 0
    all_recons = []
    all_mags = []
    with torch.no_grad():
        for p, mag in data_loader:
            p = p.to(device)
            mag = mag.to(device)
            recon, mu, logvar = model(mag, p)
            loss, mse, kld = loss_function(recon, mag, mu, logvar, beta=beta)
            total_loss += loss.item()
            total_mse  += mse.item()
            total_kld  += kld.item()
            count      += 1
            # Save for preview
            all_recons.append(recon.cpu().numpy())
            all_mags.append(mag.cpu().numpy())
    avg_loss = total_loss / count if count else float('inf')
    avg_mse  = total_mse  / count if count else float('inf')
    avg_kld  = total_kld  / count if count else float('inf')
    return avg_loss, avg_mse, avg_kld, np.concatenate(all_recons), np.concatenate(all_mags)

def plot_loss_curve(train_losses, val_losses, save_path=None):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def to_rgb_tensor(img, max_items=4):
    # img: [B, 3, H, W] or [B, H, W, 3], values in [-1, 1] or ~[-1,1]
    imgs = []
    for i in range(min(len(img), max_items)):
        arr = img[i]
        if arr.shape[0] == 3:  # [3, H, W]
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, -1, 1)
        rgb = convert_to_rgb(arr)
        imgs.append(rgb)
    imgs = np.stack(imgs)  # [N, H, W, 3]
    imgs = np.transpose(imgs, (0, 3, 1, 2))  # [N, 3, H, W]
    return torch.from_numpy(imgs).float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, required=True, help="Ścieżka do pliku HDF5 z metadanymi")
    parser.add_argument("--fields", type=str, required=True, help="Ścieżka do pliku NPZ z polami magnetyzacji")
    parser.add_argument("--beta", type=float, default=0.01, help="Waga KLD")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="model_ckpt.pth")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Część zbioru na walidację i test (łącznie, domyślnie 0.15 test + 0.15 val)")
    parser.add_argument("--no_plot", action="store_true", help="Nie pokazuj wykresu lossów (do użycia w batchach/serwerach)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimal improvement for early stopping")
    parser.add_argument("--tensorboard", action="store_true", help="Włącz tensorboard logging")
    parser.add_argument("--logdir", type=str, default="runs/exp1", help="TensorBoard log directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaling_factors = {"Aex": 1e11, "Msat": 1e-5, "Tx": 1e9, "Tz": 1e9}
    param_dim = 4

    # Wczytanie danych i podział na zbiory
    df = load_dataframe_and_fields(args.meta, args.fields)
    df_trainval, df_test = train_test_split(df, test_size=args.val_split, random_state=42)
    val_fraction_of_trainval = args.val_split / (1 - args.val_split)
    df_train, df_val = train_test_split(df_trainval, test_size=val_fraction_of_trainval, random_state=42)

    print(f"Rozmiary zbiorów: train {len(df_train)}, val {len(df_val)}, test {len(df_test)}")

    train_dataset = RealMagnetizationDatasetNew(df_train, scaling_factors)
    val_dataset = RealMagnetizationDatasetNew(df_val, scaling_factors)
    test_dataset = RealMagnetizationDatasetNew(df_test, scaling_factors)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ConditionalVAE(param_dim, args.grid_size, args.latent_dim).to(device)
    if torch.cuda.device_count() > 1:
        print("Używam DataParallel")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []

    if args.tensorboard:
        writer = SummaryWriter(log_dir=args.logdir)
    else:
        writer = None

    best_val_loss = float("inf")
    patience = args.patience
    min_delta = args.min_delta
    counter = 0
    best_state = None

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
        # Ewaluacja na walidacji
        val_loss, val_mse, val_kld, val_recons, val_mags = evaluate(model, val_loader, device, beta=args.beta)
        print(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, KLD: {val_kld:.4f})")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Logi do tensorboard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MSE/val', val_mse, epoch)
            writer.add_scalar('KLD/val', val_kld, epoch)
            # Podgląd rekonstrukcji (max 4 próbki, co 10 epok)
            if epoch % 10 == 0:
                recon_tensor = to_rgb_tensor(val_recons, max_items=4)
                mag_tensor = to_rgb_tensor(val_mags, max_items=4)
                writer.add_images("Reconstructions", recon_tensor, epoch)
                writer.add_images("Originals", mag_tensor, epoch)

        # --- EARLY STOPPING LOGIKA ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            best_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"EARLY STOPPING at epoch {epoch+1}")
                break

    # Ewaluacja na zbiorze testowym na koniec
    test_loss, test_mse, test_kld, _, _ = evaluate(model, test_loader, device, beta=args.beta)
    print(f"*** Final Test Loss: {test_loss:.4f} (MSE: {test_mse:.4f}, KLD: {test_kld:.4f})")

    # Zapisz najlepszy model
    torch.save(best_state, args.save_path)
    print(f"Zapisano najlepszy model (val_loss={best_val_loss:.5f}) do {args.save_path}")

    # Zapisz i/lub pokaż wykres lossów
    losses_dir = os.path.join(os.path.dirname(args.save_path), "losses")
    os.makedirs(losses_dir, exist_ok=True)
    np.save(os.path.join(losses_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(losses_dir, "val_losses.npy"), np.array(val_losses))

    if not args.no_plot:
        plot_loss_curve(train_losses, val_losses,
                        save_path=os.path.join(losses_dir, "loss_curve.png"))

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()


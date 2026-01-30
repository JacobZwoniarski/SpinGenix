import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def _norm_to_minus1_1(x, lo, hi):
    if hi == lo:
        return 0.0
    return float(2.0 * (x - lo) / (hi - lo) - 1.0)


def make_cond(tx, tz, device="cuda", param_ranges=None):
    """
    Zwraca cond tensor (1,2).
    Jeśli param_ranges={"Tx_val":(lo,hi),"Tz_val":(lo,hi)} -> normalizuje do [-1,1]
    """
    if param_ranges is not None:
        tx = _norm_to_minus1_1(tx, *param_ranges["Tx_val"])
        tz = _norm_to_minus1_1(tz, *param_ranges["Tz_val"])
    return torch.tensor([[tx, tz]], dtype=torch.float32, device=device)


def _empty_skips_safe(model, batch_size=1, device="cuda"):
    """
    Próbuje dostać 'puste skipy' jeśli model je wspiera.
    W razie braku - zwraca [] (a decode może wtedy nie zadziałać -> fallback w predict).
    """
    if hasattr(model, "empty_skips"):
        try:
            return model.empty_skips(batch_size=batch_size, device=device)
        except TypeError:
            try:
                return model.empty_skips()
            except Exception:
                return []
    return []


@torch.no_grad()
def _decode_one(model, cond, device="cuda", z=None, spatial_size=200):
    """
    Najpierw próbuje decode(z, empty_skips, cond).
    Jeśli to nie działa, robi fallback: forward na zerowym polu (bezpośrednio model(x,cond)).
    """
    model.eval()
    if z is None:
        latent_dim = getattr(model, "latent_dim", None)
        if latent_dim is None:
            raise AttributeError("Model has no latent_dim; cannot sample z.")
        z = torch.randn(1, latent_dim, device=device)

    # Try pure decode
    if hasattr(model, "decode"):
        try:
            skips = _empty_skips_safe(model, batch_size=1, device=device)
            out = model.decode(z, skips, cond)  # (1,3,H,W)
            return out
        except Exception:
            pass

    # Fallback: use forward with dummy field
    dummy = torch.zeros(1, 3, spatial_size, spatial_size, device=device)
    out, _, _ = model(dummy, cond)
    return out


@torch.no_grad()
def predict_phase_mz(
    model,
    Tx_vals,
    Tz_vals,
    device="cuda",
    latent_samples=8,
    spatial_size=200,
    param_ranges=None,
):
    """
    Dla list Tx_vals, Tz_vals (takiej samej długości) generuje MeanMz z modelu.
    latent_samples: ile próbek z (prior) po z, żeby uśrednić (ważne! inaczej wykres "pływa")
    Zwraca DF: Tx_val, Tz_val, MeanMz, StdMz
    """
    Tx_vals = np.asarray(Tx_vals, dtype=float).reshape(-1)
    Tz_vals = np.asarray(Tz_vals, dtype=float).reshape(-1)

    if len(Tx_vals) != len(Tz_vals):
        raise ValueError("Tx_vals and Tz_vals must have same length.")

    records = []
    model = model.to(device)
    model.eval()

    for tx, tz in zip(Tx_vals, Tz_vals):
        cond = make_cond(float(tx), float(tz), device=device, param_ranges=param_ranges)
        mz_means = []

        for _ in range(latent_samples):
            recon = _decode_one(model, cond, device=device, spatial_size=spatial_size)
            mz = recon[0, 2].detach().cpu().numpy()   # (H,W)
            mz_means.append(float(np.mean(mz)))

        records.append({
            "Tx_val": float(tx),
            "Tz_val": float(tz),
            "MeanMz": float(np.mean(mz_means)),
            "StdMz": float(np.std(mz_means)),
        })

    return pd.DataFrame(records)


def plot_phase_diagram(df, title="Phase Diagram", save_path=None, show=True):
    """
    Scatter phase diagram w Twoim stylu (jet + colorbar).
    """
    df = df.copy()
    df["Tx_nm"] = df["Tx_val"] * 1e9
    df["Tz_nm"] = df["Tz_val"] * 1e9

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("jet")
    norm = mcolors.Normalize(vmin=df["MeanMz"].min(), vmax=df["MeanMz"].max())

    scatter = sns.scatterplot(
        data=df,
        x="Tx_nm",
        y="Tz_nm",
        hue="MeanMz",
        palette=cmap,
        hue_norm=norm,
        s=50,
        legend=False
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=scatter.axes, label="Mean Mz")

    plt.xlabel("Tx (nm)")
    plt.ylabel("Tz (nm)")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def load_phase_hdf5(hdf5_path, key="data"):
    """
    Ładuje fazę z symulacji (np. phase_classification_results_abs.h5).
    """
    df = pd.read_hdf(hdf5_path, key=key)
    if "Tx_val" not in df.columns or "Tz_val" not in df.columns:
        raise ValueError("Expected columns Tx_val and Tz_val in HDF5.")
    if "MeanMz" not in df.columns:
        raise ValueError("Expected column MeanMz in HDF5.")
    return df


def predict_phase_on_grid(
    model,
    Tx_range=(10e-9, 100e-9),
    Tz_range=(10e-9, 100e-9),
    grid_points=40,
    device="cuda",
    latent_samples=8,
    spatial_size=200,
    param_ranges=None,
):
    Tx = np.linspace(Tx_range[0], Tx_range[1], grid_points)
    Tz = np.linspace(Tz_range[0], Tz_range[1], grid_points)
    Tx_grid, Tz_grid = np.meshgrid(Tx, Tz, indexing="xy")

    df = predict_phase_mz(
        model,
        Tx_grid.reshape(-1),
        Tz_grid.reshape(-1),
        device=device,
        latent_samples=latent_samples,
        spatial_size=spatial_size,
        param_ranges=param_ranges,
    )
    return df, Tx_grid, Tz_grid
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from SpinGenix.active_learning.visualization import convert_to_rgb


@torch.no_grad()
def predict_phase_mz(model, Tx_grid, Tz_grid, device="cuda"):
    """
    Generate MeanMz for all grid points using the CVAE decoder.
    Returns:
        DataFrame with Tx_val, Tz_val, MeanMz
    """
    Tx_flat = Tx_grid.reshape(-1)
    Tz_flat = Tz_grid.reshape(-1)

    records = []

    for tx, tz in zip(Tx_flat, Tz_flat):
        cond = torch.tensor([[tx, tz]], dtype=torch.float32, device=device)

        # Feed decoder with prior z
        z = torch.randn(1, model.latent_dim, device=device)
        recon = model.decode(z, model.empty_skips(), cond)  # shape (1,3,H,W)
        recon = recon[0].cpu().numpy()  # (3, H, W)
        mz = recon[2]                   # (H,W)

        mean_mz = float(np.mean(mz))

        records.append({
            "Tx_val": tx,
            "Tz_val": tz,
            "MeanMz": mean_mz
        })

    return pd.DataFrame(records)


def plot_phase_diagram(df):
    """
    Scatter phase diagram identical to your baseline version.
    """
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
    plt.title("Model-Predicted Phase Diagram")

    plt.tight_layout()
    plt.show()
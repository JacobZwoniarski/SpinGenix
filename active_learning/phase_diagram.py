import os
import getpass
import tempfile
import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@torch.no_grad()
def predict_phase_mz(model, Tx_grid, Tz_grid, device="cuda", param_normalizer=None):
    """
    Generate MeanMz for all grid points using the CVAE decoder.
    Returns:
        DataFrame with Tx_val, Tz_val, MeanMz_signed, MeanMz_abs, MeanMz
    """
    records = []
    H = getattr(model, "spatial_size", 200)
    W = getattr(model, "spatial_size", 200)
    warned_dummy = False

    model.to(device)
    model.eval()

    for tx in Tx_grid:
        for tz in Tz_grid:
            cond_values = np.array([tx, tz], dtype=np.float64)
            if param_normalizer is not None:
                cond_values = param_normalizer.transform(cond_values)
            cond = torch.tensor([cond_values], dtype=torch.float32, device=device)
            if hasattr(model, "sample"):
                recon = model.sample(cond)
            else:
                if not warned_dummy:
                    warnings.warn(
                        "Model has no sample(params) method; using a zero-field "
                        "reconstruction path for diagnostics only.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_dummy = True
                dummy = torch.zeros((1, 3, H, W), dtype=torch.float32, device=device)
                recon, _, _ = model(dummy, cond)
            recon = recon[0].cpu().numpy()  # (3, H, W)
            mz = recon[2]
            mean_mz_abs = float(np.mean(np.abs(mz)))

            records.append({
                "Tx_val": tx,
                "Tz_val": tz,
                "MeanMz_signed": float(np.mean(mz)),
                "MeanMz_abs": mean_mz_abs,
                "MeanMz": mean_mz_abs,
            })

    return pd.DataFrame(records)


def _default_value_column(df):
    if "MeanMz_abs" in df.columns:
        return "MeanMz_abs"
    if "MeanMz" in df.columns:
        return "MeanMz"
    raise KeyError("Expected a phase value column named 'MeanMz_abs' or 'MeanMz'.")


def plot_phase_diagram(
    df,
    value_col=None,
    title="Phase Diagram",
    colorbar_label="Mean Mz",
    cmap="jet",
    ax=None,
    save_path=None,
    show=False,
):
    """
    Scatter phase diagram for Tx/Tz data. Defaults to MeanMz_abs when available,
    matching the historical absolute-Mz validation plot.

    Returns:
        fig, ax
    """
    value_col = value_col or _default_value_column(df)
    plot_df = df.copy()
    if "Tx_nm" not in plot_df.columns:
        plot_df["Tx_nm"] = plot_df["Tx_val"] * 1e9
    if "Tz_nm" not in plot_df.columns:
        plot_df["Tz_nm"] = plot_df["Tz_val"] * 1e9

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    cmap = plt.get_cmap(cmap)
    values = plot_df[value_col].astype(float)
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

    scatter = ax.scatter(
        plot_df["Tx_nm"],
        plot_df["Tz_nm"],
        c=values,
        cmap=cmap,
        norm=norm,
        s=50,
    )

    fig.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_xlabel("Tx (nm)")
    ax.set_ylabel("Tz (nm)")
    ax.set_title(title)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_dataset_phase_diagram(df, save_path=None, show=False):
    return plot_phase_diagram(
        df,
        value_col="MeanMz_abs" if "MeanMz_abs" in df.columns else None,
        title="Phase Diagram",
        colorbar_label="Mean Mz",
        save_path=save_path,
        show=show,
    )

import os
import getpass
import tempfile

import torch
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri

try:
    from scipy.interpolate import griddata
except Exception:  # pragma: no cover - scipy is part of the project env
    griddata = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - scipy is part of the project env
    gaussian_filter = None


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

    model.to(device)
    model.eval()

    for tx in Tx_grid:
        for tz in Tz_grid:
            cond_values = np.array([tx, tz], dtype=np.float64)
            if param_normalizer is not None:
                cond_values = param_normalizer.transform(cond_values)
            cond_values = np.asarray(cond_values, dtype=np.float32)[None, :]
            cond = torch.from_numpy(cond_values).to(device)
            if hasattr(model, "sample"):
                recon = model.sample(cond)
            else:
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


def _phase_axes_dataframe(df):
    plot_df = df.copy()
    if "Tx_nm" not in plot_df.columns:
        plot_df["Tx_nm"] = plot_df["Tx_val"] * 1e9
    if "Tz_nm" not in plot_df.columns:
        plot_df["Tz_nm"] = plot_df["Tz_val"] * 1e9
    return plot_df


def _value_norm(values, value_col):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Phase diagram values contain no finite entries.")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.01, 1e-6)
        return mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)

    signed_name = "signed" in str(value_col).lower()
    crosses_zero = vmin < 0 < vmax
    if signed_name or crosses_zero:
        span = max(abs(vmin), abs(vmax))
        return mcolors.TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _default_cmap(value_col, cmap):
    if cmap is not None:
        return cmap
    if "signed" in str(value_col).lower():
        return "coolwarm"
    return "viridis"


def _axis_limits(values):
    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.02, 1.0)
        return lower - pad, upper + pad
    return lower, upper


def _plot_landscape(
    ax,
    x,
    y,
    values,
    cmap,
    norm,
    grid_resolution=240,
    interpolation="linear",
    levels=48,
    fill_nearest=True,
    smooth_sigma=1.0,
):
    if len(values) < 3:
        return None

    x_unique = np.unique(x[np.isfinite(x)])
    y_unique = np.unique(y[np.isfinite(y)])
    if x_unique.size < 2 or y_unique.size < 2:
        return None

    if griddata is not None:
        xi = np.linspace(float(np.min(x)), float(np.max(x)), grid_resolution)
        yi = np.linspace(float(np.min(y)), float(np.max(y)), grid_resolution)
        grid_x, grid_y = np.meshgrid(xi, yi)
        method = interpolation
        if method == "cubic" and len(values) < 16:
            method = "linear"
        try:
            grid_z = griddata((x, y), values, (grid_x, grid_y), method=method)
            if grid_z is not None and np.isfinite(grid_z).any():
                if fill_nearest and np.isnan(grid_z).any():
                    nearest = griddata((x, y), values, (grid_x, grid_y), method="nearest")
                    if nearest is not None:
                        grid_z = np.where(np.isnan(grid_z), nearest, grid_z)
                if smooth_sigma and smooth_sigma > 0 and gaussian_filter is not None:
                    grid_z = gaussian_filter(grid_z, sigma=float(smooth_sigma))
                return ax.imshow(
                    grid_z,
                    extent=(float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))),
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                    interpolation="bicubic",
                    aspect="auto",
                )
        except Exception:
            pass

    try:
        triangulation = mtri.Triangulation(x, y)
        return ax.tricontourf(
            triangulation,
            values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            antialiased=True,
        )
    except Exception:
        return None


def plot_phase_diagram(
    df,
    value_col=None,
    title="Phase Diagram",
    colorbar_label="Mean Mz",
    cmap=None,
    style="landscape",
    overlay_points=True,
    point_size=22,
    grid_resolution=300,
    interpolation="linear",
    levels=48,
    fill_nearest=True,
    smooth_sigma=1.0,
    ax=None,
    save_path=None,
    show=False,
):
    """
    Plot a Tx/Tz phase diagram. By default this renders a continuous interpolated
    landscape with source points overlaid, which makes phase transitions easier
    to see than a point cloud while still showing where data actually exists.

    Returns:
        fig, ax
    """
    value_col = value_col or _default_value_column(df)
    plot_df = _phase_axes_dataframe(df)
    plot_df = plot_df.dropna(subset=["Tx_nm", "Tz_nm", value_col])
    if plot_df.empty:
        raise ValueError("Phase diagram dataframe has no plottable rows.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    cmap = plt.get_cmap(_default_cmap(value_col, cmap))
    values = plot_df[value_col].astype(float)
    norm = _value_norm(values, value_col)

    x = plot_df["Tx_nm"].to_numpy(dtype=float)
    y = plot_df["Tz_nm"].to_numpy(dtype=float)
    value_array = values.to_numpy(dtype=float)

    mappable = None
    if style in {"landscape", "surface", "smooth"}:
        mappable = _plot_landscape(
            ax,
            x,
            y,
            value_array,
            cmap=cmap,
            norm=norm,
            grid_resolution=grid_resolution,
            interpolation=interpolation,
            levels=levels,
            fill_nearest=fill_nearest,
            smooth_sigma=smooth_sigma,
        )
    elif style != "scatter":
        raise ValueError("style must be one of: 'landscape', 'surface', 'smooth', 'scatter'")

    should_scatter = style == "scatter" or overlay_points or mappable is None
    if should_scatter:
        scatter = ax.scatter(
            x,
            y,
            c=value_array,
            cmap=cmap,
            norm=norm,
            s=point_size,
            edgecolors="black" if style != "scatter" else "none",
            linewidths=0.25 if style != "scatter" else 0.0,
            alpha=0.6 if style != "scatter" else 1.0,
        )
        if mappable is None:
            mappable = scatter

    fig.colorbar(mappable, ax=ax, label=colorbar_label)
    ax.set_xlabel("Tx (nm)")
    ax.set_ylabel("Tz (nm)")
    ax.set_title(title)
    ax.set_xlim(*_axis_limits(x))
    ax.set_ylim(*_axis_limits(y))

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
        colorbar_label="|Mean Mz|",
        save_path=save_path,
        show=show,
    )

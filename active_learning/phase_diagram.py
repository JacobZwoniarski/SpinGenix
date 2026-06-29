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
    Generate scalar phase metrics for all grid points using a params -> field model.

    For multi-parameter normalizers, Tx/Tz are swept when present and all
    remaining parameters are frozen at the midpoint of their training range.
    Older reconstruction CVAE models are still supported only as a debug
    fallback by feeding a zero field.
    """
    records = []
    H = getattr(model, "spatial_size", 200)
    W = getattr(model, "spatial_size", 200)

    model.to(device)
    model.eval()

    if param_normalizer is not None:
        columns = tuple(param_normalizer.param_columns)
        column_to_idx = {column: idx for idx, column in enumerate(columns)}
        tx_idx = column_to_idx.get("Tx_val", 0)
        tz_idx = column_to_idx.get("Tz_val", 1 if len(columns) > 1 else 0)
        base = 0.5 * (np.asarray(param_normalizer.mins) + np.asarray(param_normalizer.maxs))
    else:
        columns = ("Tx_val", "Tz_val")
        tx_idx = 0
        tz_idx = 1
        base = np.zeros(2, dtype=np.float64)

    for tx in Tx_grid:
        for tz in Tz_grid:
            cond_values = base.copy()
            cond_values[tx_idx] = tx
            cond_values[tz_idx] = tz
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
                "Tx_val": float(tx),
                "Tz_val": float(tz),
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
        return "RdBu_r"
    return "cividis"


def _prepare_cmap(value_col, cmap):
    resolved = plt.get_cmap(_default_cmap(value_col, cmap))
    try:
        resolved = resolved.copy()
    except AttributeError:
        pass
    resolved.set_bad("#f6f7f5", alpha=0.0)
    return resolved


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
    grid_resolution=220,
    interpolation="linear",
    levels=32,
    fill_nearest=False,
    smooth_sigma=0.0,
    xlim_nm=None,
    ylim_nm=None,
):
    if len(values) < 3:
        return None

    x_unique = np.unique(x[np.isfinite(x)])
    y_unique = np.unique(y[np.isfinite(y)])
    if x_unique.size < 2 or y_unique.size < 2:
        return None

    if x_unique.size * y_unique.size == len(values):
        grid_z = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
        x_index = {value: idx for idx, value in enumerate(x_unique)}
        y_index = {value: idx for idx, value in enumerate(y_unique)}
        has_duplicate = False
        for xi, yi, zi in zip(x, y, values):
            row = y_index[yi]
            col = x_index[xi]
            if np.isfinite(grid_z[row, col]):
                has_duplicate = True
                break
            grid_z[row, col] = zi
        if not has_duplicate:
            return ax.pcolormesh(
                x_unique,
                y_unique,
                np.ma.masked_invalid(grid_z),
                shading="auto",
                cmap=cmap,
                norm=norm,
                antialiased=False,
                rasterized=True,
            )

    if griddata is not None:
        x_lower, x_upper = xlim_nm if xlim_nm is not None else (float(np.min(x)), float(np.max(x)))
        y_lower, y_upper = ylim_nm if ylim_nm is not None else (float(np.min(y)), float(np.max(y)))
        xi = np.linspace(float(x_lower), float(x_upper), grid_resolution)
        yi = np.linspace(float(y_lower), float(y_upper), grid_resolution)
        grid_x, grid_y = np.meshgrid(xi, yi)
        method = interpolation
        if method not in {"nearest", "linear", "cubic"}:
            method = "linear"
        if method == "cubic" and len(values) < 16:
            method = "linear"
        try:
            grid_z = griddata((x, y), values, (grid_x, grid_y), method=method)
            if grid_z is not None and np.isfinite(grid_z).any():
                if fill_nearest and np.isnan(grid_z).any():
                    nearest = griddata((x, y), values, (grid_x, grid_y), method="nearest")
                    if nearest is not None:
                        grid_z = np.where(np.isnan(grid_z), nearest, grid_z)
                if (
                    smooth_sigma
                    and smooth_sigma > 0
                    and gaussian_filter is not None
                    and np.isfinite(grid_z).all()
                ):
                    grid_z = gaussian_filter(grid_z, sigma=float(smooth_sigma))
                return ax.imshow(
                    np.ma.masked_invalid(grid_z),
                    extent=(float(x_lower), float(x_upper), float(y_lower), float(y_upper)),
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                    aspect="auto",
                )
        except Exception:
            pass

    try:
        triangulation = mtri.Triangulation(x, y)
        return ax.tripcolor(
            triangulation,
            values,
            shading="flat",
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
    point_size=30,
    grid_resolution=220,
    interpolation="linear",
    levels=32,
    fill_nearest=False,
    smooth_sigma=0.0,
    xlim_nm=None,
    ylim_nm=None,
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

    cmap = _prepare_cmap(value_col, cmap)
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
            xlim_nm=xlim_nm,
            ylim_nm=ylim_nm,
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
            edgecolors="#1f2528",
            linewidths=0.35,
            alpha=0.82 if style != "scatter" else 0.94,
        )
        if mappable is None:
            mappable = scatter

    fig.colorbar(mappable, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    ax.set_xlabel("Tx (nm)")
    ax.set_ylabel("Tz (nm)")
    ax.set_title(title)
    ax.set_xlim(*(xlim_nm if xlim_nm is not None else _axis_limits(x)))
    ax.set_ylim(*(ylim_nm if ylim_nm is not None else _axis_limits(y)))
    ax.set_facecolor("#f6f7f5")
    ax.grid(color="#d7ddd8", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def _padded_limits(limits, pad_fraction=0.02):
    if limits is None:
        return None
    lower, upper = float(limits[0]), float(limits[1])
    pad = max((upper - lower) * pad_fraction, 1e-9)
    return lower - pad, upper + pad


def plot_dataset_phase_diagram(
    df,
    save_path=None,
    show=False,
    tx_range_nm=None,
    tz_range_nm=None,
):
    return plot_phase_diagram(
        df,
        value_col="MeanMz_abs" if "MeanMz_abs" in df.columns else None,
        title="Dataset Phase Diagram",
        colorbar_label="|Mean Mz|",
        cmap="turbo",
        style="scatter",
        overlay_points=False,
        point_size=42,
        xlim_nm=_padded_limits(tx_range_nm),
        ylim_nm=_padded_limits(tz_range_nm),
        save_path=save_path,
        show=show,
    )


def plot_dataset_phase_comparison(
    df,
    save_path=None,
    show=False,
    tx_range_nm=None,
    tz_range_nm=None,
    value_col=None,
):
    value_col = value_col or ("MeanMz_abs" if "MeanMz_abs" in df.columns else None)
    plot_df = _phase_axes_dataframe(df).dropna(subset=["Tx_nm", "Tz_nm", value_col or _default_value_column(df)])
    value_col = value_col or _default_value_column(plot_df)
    values = plot_df[value_col].astype(float).to_numpy(dtype=float)
    norm = _value_norm(values, value_col)
    cmap = _prepare_cmap(value_col, "turbo")
    xlim = _padded_limits(tx_range_nm)
    ylim = _padded_limits(tz_range_nm)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)

    scatter = axes[0].scatter(
        plot_df["Tx_nm"],
        plot_df["Tz_nm"],
        c=values,
        cmap=cmap,
        norm=norm,
        s=44,
        edgecolors="#1f2528",
        linewidths=0.35,
        alpha=0.94,
    )
    axes[0].set_title(f"Observed samples (n={len(plot_df)})")

    mappable = _plot_landscape(
        axes[1],
        plot_df["Tx_nm"].to_numpy(dtype=float),
        plot_df["Tz_nm"].to_numpy(dtype=float),
        values,
        cmap=cmap,
        norm=norm,
        grid_resolution=260,
        interpolation="linear",
        fill_nearest=True,
        smooth_sigma=1.1,
        xlim_nm=tx_range_nm,
        ylim_nm=tz_range_nm,
    )
    if mappable is None:
        mappable = scatter
    axes[1].scatter(
        plot_df["Tx_nm"],
        plot_df["Tz_nm"],
        facecolors="none",
        edgecolors="#1f2528",
        s=18,
        linewidths=0.35,
        alpha=0.45,
    )
    axes[1].set_title("Interpolated view")

    for ax in axes:
        ax.set_xlabel("Tx (nm)")
        ax.set_ylabel("Tz (nm)")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_facecolor("#f6f7f5")
        ax.grid(color="#d7ddd8", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

    fig.colorbar(mappable, ax=axes.ravel().tolist(), label="|Mean Mz|", fraction=0.046, pad=0.035)
    fig.suptitle("Dataset Phase Diagram: Observed vs Interpolated", y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes

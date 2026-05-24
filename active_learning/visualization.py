import numpy as np
import colorsys
import getpass
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

CHANNELS = ("Mx", "My", "Mz")


# =======================================================================
# HSL-based visualization (your notebook version)
# =======================================================================

def hsl_pixel_to_rgb(h, s, l):
    """Convert single pixel HSL → RGB."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)  # NOTE: colorsys uses HLS, not HSL (order: h, l, s)
    return r, g, b


def convert_hsl_to_rgb(hsl_img):
    """
    Convert full HSL image (H,W,3) → RGB float32 image (H,W,3).
    Values are assumed in [0,1].
    """
    H, W, _ = hsl_img.shape
    rgb = np.zeros_like(hsl_img, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            h, s, l = hsl_img[i, j]
            rgb[i, j] = hsl_pixel_to_rgb(h, s, l)

    return rgb


def magnetization_to_hsl_rgb(arr):
    """
    Convert magnetization field (H,W,3) with values in [-1,1]
    into HSL → RGB representation.

    Hue      = angle(mx + i my)
    Saturation = in-plane magnitude
    Lightness  = (mz+1)/2

    Returns: RGB image in [0,1], shape (H,W,3)
    """
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    u = arr[:, :, 0]     # mx
    v = arr[:, :, 1]     # my
    w = arr[:, :, 2]     # mz

    # Hue from phase angle
    H = np.angle(u + 1j * v)
    H = (H + np.pi) / (2 * np.pi)   # map [-π, π] → [0,1]

    # Saturation must be in-plane only. Using full |m| makes nearly uniform
    # +/-Mz fields look falsely rainbow-colored because |m| ~= 1 everywhere.
    S = np.sqrt(u*u + v*v)
    S = np.clip(S, 0, 1)

    # Lightness = scaled mz
    L = (w + 1) / 2.0
    L = np.clip(L, 0, 1)

    hsl = np.stack([H, S, L], axis=-1)
    return convert_hsl_to_rgb(hsl)


# =======================================================================
# Component-based visualization
# =======================================================================

def magnetization_to_component_rgb(arr):
    """
    Simple scientific representation:
        R = normalized mx
        G = normalized my
        B = normalized mz
    """
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    rgb = (arr + 1) / 2.0
    rgb = np.clip(rgb, 0, 1)
    return rgb.astype(np.float32)


# =======================================================================
# Unified wrapper
# =======================================================================

def convert_to_rgb(field, mode="hsl"):
    """
    Convert magnetization array (H,W,3) to RGB using mode:
        - "hsl"
        - "components"
    """
    if mode == "hsl":
        return magnetization_to_hsl_rgb(field)
    elif mode == "components":
        return magnetization_to_component_rgb(field)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'hsl' or 'components'.")


def _to_hwc_array(field):
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()

    field = np.asarray(field)
    if field.ndim == 4 and field.shape[0] == 1:
        field = field[0]
    if field.ndim != 3:
        raise ValueError(f"Expected field shape (3,H,W) or (H,W,3), got {field.shape}")
    if field.shape[0] == 3 and field.shape[-1] != 3:
        field = np.transpose(field, (1, 2, 0))
    if field.shape[-1] != 3:
        raise ValueError(f"Expected 3 magnetization channels, got {field.shape}")

    return np.nan_to_num(field.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=-1.0)


# =======================================================================
# Reconstruction visualization
# =======================================================================

def visualize_reconstruction(original, predicted, Tx, Tz, mode="hsl", save_path=None, show=False):
    """
    Build a side-by-side reconstruction figure for a single sample.

    original, predicted : tensors or numpy arrays in shape (3,H,W) or (H,W,3)
    Tx, Tz : param values (floats)

    Returns:
        fig, axes
    """
    if mode in {"component_panels", "component_panel", "panels"}:
        return visualize_reconstruction_components(
            original,
            predicted,
            Tx,
            Tz,
            save_path=save_path,
            show=show,
        )

    original = _to_hwc_array(original)
    predicted = _to_hwc_array(predicted)

    rgb_orig = convert_to_rgb(original, mode=mode)
    rgb_pred = convert_to_rgb(predicted, mode=mode)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].set_title(f"Original\nTx={Tx*1e9:.1f} nm, Tz={Tz*1e9:.1f} nm")
    axes[0].imshow(rgb_orig)
    axes[0].axis("off")

    axes[1].set_title(f"Reconstruction ({mode})\nTx={Tx*1e9:.1f} nm, Tz={Tz*1e9:.1f} nm")
    axes[1].imshow(rgb_pred)
    axes[1].axis("off")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def visualize_reconstruction_components(
    original,
    predicted,
    Tx,
    Tz,
    save_path=None,
    show=False,
    diff_percentile=99.0,
):
    """
    Scientific reconstruction diagnostic: target, prediction, and residual for
    each magnetization component. Component colors are fixed to [-1, 1], so
    panels from different samples stay comparable.
    """
    original = np.clip(_to_hwc_array(original), -1.0, 1.0)
    predicted = np.clip(_to_hwc_array(predicted), -1.0, 1.0)
    diff = predicted - original

    finite_abs_diff = np.abs(diff[np.isfinite(diff)])
    if finite_abs_diff.size:
        diff_limit = float(np.percentile(finite_abs_diff, diff_percentile))
        diff_limit = max(diff_limit, 0.05)
    else:
        diff_limit = 1.0

    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
    column_titles = ("Target", "Prediction", "Difference")
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    component_mappable = None
    diff_mappable = None
    for row, channel in enumerate(CHANNELS):
        axes[row, 0].set_ylabel(channel, rotation=0, labelpad=22, va="center")
        component_mappable = axes[row, 0].imshow(
            original[:, :, row],
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[row, 1].imshow(
            predicted[:, :, row],
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        diff_mappable = axes[row, 2].imshow(
            diff[:, :, row],
            cmap="RdBu_r",
            vmin=-diff_limit,
            vmax=diff_limit,
            interpolation="nearest",
        )
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(f"Tx={Tx*1e9:.1f} nm, Tz={Tz*1e9:.1f} nm", y=1.02)
    if component_mappable is not None:
        fig.colorbar(
            component_mappable,
            ax=axes[:, :2].ravel().tolist(),
            shrink=0.75,
            label="m",
        )
    if diff_mappable is not None:
        fig.colorbar(
            diff_mappable,
            ax=axes[:, 2].ravel().tolist(),
            shrink=0.75,
            label="prediction - target",
        )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes

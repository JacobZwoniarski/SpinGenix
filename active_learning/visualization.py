import numpy as np
import colorsys
import getpass
import os
import tempfile
import matplotlib

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch


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
    Saturation = magnitude
    Lightness  = (mz+1)/2

    Returns: RGB image in [0,1], shape (H,W,3)
    """
    u = arr[:, :, 0]     # mx
    v = arr[:, :, 1]     # my
    w = arr[:, :, 2]     # mz

    # Hue from phase angle
    H = np.angle(u + 1j * v)
    H = (H + np.pi) / (2 * np.pi)   # map [-π, π] → [0,1]

    # Saturation = magnitude
    S = np.sqrt(u*u + v*v + w*w)
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

    return field


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

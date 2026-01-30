import os
import numpy as np
import matplotlib.pyplot as plt
import torch


# ======================================================================
# FAST (vectorized) HSL -> RGB
# ======================================================================

def _hue2rgb(p, q, t):
    t = np.mod(t, 1.0)
    a = (t < 1/6)
    b = (t >= 1/6) & (t < 1/2)
    c = (t >= 1/2) & (t < 2/3)

    out = np.empty_like(t, dtype=np.float32)
    out[a] = p[a] + (q[a] - p[a]) * 6.0 * t[a]
    out[b] = q[b]
    out[c] = p[c] + (q[c] - p[c]) * (2/3 - t[c]) * 6.0
    out[~(a | b | c)] = p[~(a | b | c)]
    return out


def hsl_to_rgb(h, s, l):
    """
    Vectorized HSL -> RGB.
    h,s,l arrays in [0,1], returns rgb in [0,1], shape (...,3).
    """
    h = h.astype(np.float32)
    s = s.astype(np.float32)
    l = l.astype(np.float32)

    rgb = np.zeros((*h.shape, 3), dtype=np.float32)

    # achromatic
    mask0 = (s <= 1e-12)
    rgb[mask0, 0] = l[mask0]
    rgb[mask0, 1] = l[mask0]
    rgb[mask0, 2] = l[mask0]

    # chromatic
    mask = ~mask0
    if np.any(mask):
        hh = h[mask]
        ss = s[mask]
        ll = l[mask]

        q = np.where(ll < 0.5, ll * (1 + ss), ll + ss - ll * ss).astype(np.float32)
        p = (2 * ll - q).astype(np.float32)

        r = _hue2rgb(p, q, hh + 1/3)
        g = _hue2rgb(p, q, hh)
        b = _hue2rgb(p, q, hh - 1/3)

        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b

    return rgb


# ======================================================================
# Magnetization -> RGB
# ======================================================================

def magnetization_to_hsl_rgb(arr_hw3: np.ndarray) -> np.ndarray:
    """
    arr_hw3: (H,W,3) in [-1,1] -> RGB (H,W,3) in [0,1]
    Hue = angle(mx + i my), Sat = norm, Light = (mz+1)/2
    """
    u = arr_hw3[:, :, 0]
    v = arr_hw3[:, :, 1]
    w = arr_hw3[:, :, 2]

    H = np.angle(u + 1j * v).astype(np.float32)
    H = (H + np.pi) / (2 * np.pi)  # [0,1]

    S = np.sqrt(u*u + v*v + w*w).astype(np.float32)
    S = np.clip(S, 0, 1)

    L = ((w + 1) / 2.0).astype(np.float32)
    L = np.clip(L, 0, 1)

    return hsl_to_rgb(H, S, L)


def magnetization_to_component_rgb(arr_hw3: np.ndarray) -> np.ndarray:
    rgb = (arr_hw3 + 1) / 2.0
    return np.clip(rgb, 0, 1).astype(np.float32)


def convert_to_rgb(field, mode="hsl"):
    """
    field: (H,W,3) in [-1,1]
    mode: 'hsl' or 'components'
    """
    if mode == "hsl":
        return magnetization_to_hsl_rgb(field)
    if mode == "components":
        return magnetization_to_component_rgb(field)
    raise ValueError(f"Unknown mode '{mode}' (use 'hsl' or 'components').")


# ======================================================================
# Reconstruction visualization
# ======================================================================

def _to_hw3(x):
    # accepts torch or np, (3,H,W) or (H,W,3)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x.shape}")
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return x.astype(np.float32)


def visualize_reconstruction(original, predicted, Tx, Tz, mode="hsl", title_prefix=""):
    orig = _to_hw3(original)
    pred = _to_hw3(predicted)

    rgb_orig = convert_to_rgb(orig, mode=mode)
    rgb_pred = convert_to_rgb(pred, mode=mode)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"{title_prefix}Original\nTx={Tx*1e9:.1f} nm, Tz={Tz*1e9:.1f} nm")
    plt.imshow(rgb_orig)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"{title_prefix}Reconstruction ({mode})\nTx={Tx*1e9:.1f} nm, Tz={Tz*1e9:.1f} nm")
    plt.imshow(rgb_pred)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_reconstruction_images(
    model,
    dataset,
    out_dir="results/reconstructions",
    max_samples=16,
    mode="hsl",
    device="cuda",
    seed=0,
):
    """
    Zapisuje side-by-side (GT vs recon) dla próbek z datasetu.
    Dataset może zwracać:
      - (field, params) albo
      - (field, params, meta)
    """
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    rng = np.random.default_rng(seed)
    n = len(dataset)
    if n == 0:
        print("[viz] Dataset empty, nothing to save.")
        return

    idxs = rng.choice(n, size=min(max_samples, n), replace=False)

    for k, idx in enumerate(idxs):
        item = dataset[idx]
        if len(item) == 2:
            field, params = item
            meta = {}
        else:
            field, params, meta = item

        field_b = field.unsqueeze(0).to(device)      # (1,3,H,W)
        params_b = params.unsqueeze(0).to(device)    # (1,P)

        with torch.no_grad():
            recon, _, _ = model(field_b, params_b)

        # Tx/Tz z meta (preferowane) lub z params (jeśli jest tylko 2D i w metrach)
        Tx = float(meta.get("Tx_val", params[0].item()))
        Tz = float(meta.get("Tz_val", params[1].item()))

        gt_hw3 = _to_hw3(field)
        rc_hw3 = _to_hw3(recon[0])

        rgb_gt = convert_to_rgb(gt_hw3, mode=mode)
        rgb_rc = convert_to_rgb(rc_hw3, mode=mode)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(rgb_gt); ax1.axis("off")
        ax1.set_title(f"GT\nTx={Tx*1e9:.1f} nm, Tz={Tz*1e9:.1f} nm")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(rgb_rc); ax2.axis("off")
        ax2.set_title(f"Recon ({mode})\nTx={Tx*1e9:.1f} nm, Tz={Tz*1e9:.1f} nm")

        fig.tight_layout()
        path = os.path.join(out_dir, f"recon_{k:03d}_Tx_{Tx:.3e}_Tz_{Tz:.3e}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

    print(f"[viz] Saved {len(idxs)} reconstructions to: {out_dir}")

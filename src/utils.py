# src/utils.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class RealMagnetizationDatasetNew(Dataset):
    def __init__(self, df, scaling_factors):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.scaling_factors = scaling_factors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        Aex_scaled = row["Aex"]   * self.scaling_factors["Aex"]
        Msat_scaled = row["Msat"] * self.scaling_factors["Msat"]
        Tx_scaled   = row["Tx_val"] * self.scaling_factors["Tx"]
        Tz_scaled   = row["Tz_val"] * self.scaling_factors["Tz"]
        p = np.array([Aex_scaled, Msat_scaled, Tx_scaled, Tz_scaled], dtype=np.float32)
        mag_field = row["field"]
        if mag_field.ndim == 4:
            mag_field = mag_field[0]
        if mag_field.shape[-1] == 3:
            mag_field = np.transpose(mag_field, (2, 0, 1))
        return torch.FloatTensor(p), torch.FloatTensor(mag_field)

def load_dataframe_and_fields(meta_in, fields_in):
    df = pd.read_hdf(meta_in, key="data")
    for col in ["Aex", "Msat", "Tx_val", "Tz_val"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fields_npz = np.load(fields_in)
    fields_list = []
    for i in range(len(df)):
        arr = fields_npz[str(i)]
        fields_list.append(arr)
    df["field"] = fields_list
    return df
# --- Klasyfikacja tekstur, liczba skyrmionowa ---

def compute_skyrmion_number(mx, my, mz, dx=0.1, dy=0.1):
    dmx_dy, dmx_dx = np.gradient(mx, dy, dx)
    dmy_dy, dmy_dx = np.gradient(my, dy, dx)
    dmz_dy, dmz_dx = np.gradient(mz, dy, dx)
    cross_x = dmy_dx * dmz_dy - dmz_dx * dmy_dy
    cross_y = dmz_dx * dmx_dy - dmx_dx * dmz_dy
    cross_z = dmx_dx * dmy_dy - dmy_dx * dmx_dy
    q = mx * cross_x + my * cross_y + mz * cross_z
    Q = np.sum(q) * dx * dy / (4 * np.pi)
    return Q

def is_vortex(mx, my, mz, dx=1.0, dy=1.0, threshold=0.1):
    mx_np, my_np, mz_np = np.asarray(mx), np.asarray(my), np.asarray(mz)
    Q = compute_skyrmion_number(mx_np, my_np, np.abs(mz_np), dx, dy)
    if abs(Q) > threshold:
        return True, Q
    else:
        return False, Q

def classify_texture_reconstructed(mx, my, mz,
                                  thr_outplane=0.65,
                                  thr_b=0.2,
                                  threshold_vortex=0.1):
    mx_r = np.asarray(mx)
    my_r = np.asarray(my)
    mz_r = np.asarray(mz)
    mean_mx_recon = np.mean(mx_r)
    mean_my_recon = np.mean(my_r)
    mean_mz_recon = np.mean(mz_r)
    b_recon = np.sqrt(mean_mx_recon**2 + mean_my_recon**2)
    is_vortex_recon, Q_recon = is_vortex(mx_r, my_r, mz_r, threshold=threshold_vortex)
    if np.abs(mean_mz_recon) > thr_outplane:
        state = "out-of-plane"
    elif is_vortex_recon:
        state = "vortex"
    elif b_recon > thr_b:
        state = "in-plane"
    else:
        state = "domain-wall"
    return state, mean_mx_recon, mean_my_recon, mean_mz_recon, b_recon, Q_recon

# --- HSL do RGB (dla wizualizacji) ---
import colorsys

def hsl2rgb_pixel(h, s, l):
    return colorsys.hls_to_rgb(h, l, s)

def hsl2rgb(hsl):
    H_img, W_img, _ = hsl.shape
    rgb = np.zeros_like(hsl)
    for i in range(H_img):
        for j in range(W_img):
            h_val = hsl[i, j, 0]
            s_val = hsl[i, j, 1]
            l_val = hsl[i, j, 2]
            try:
                r, g, b = colorsys.hls_to_rgb(h_val, l_val, s_val)
                rgb[i, j, :] = [r, g, b]
            except Exception:
                rgb[i, j, :] = [1, 0, 1] # magenta dla błędu
    return np.clip(rgb, 0, 1)

def convert_to_rgb(arr):
    arr = np.asarray(arr)
    if arr.shape[-1] != 3:
        raise ValueError("Input array must have shape (H, W, 3)")
    u = arr[:, :, 0]
    v = arr[:, :, 1]
    w = arr[:, :, 2]
    H = (np.angle(u + 1j * v) + np.pi) / (2 * np.pi)
    H = np.clip(H, 0, 1)
    S = np.sqrt(u**2 + v**2 + w**2)
    S = np.clip(S, 0, 1)
    L = (w + 1.0) / 2.0
    L = np.clip(L, 0, 1)
    hsl = np.stack((H, S, L), axis=-1)
    rgb = hsl2rgb(hsl)
    return rgb

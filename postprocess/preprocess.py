from __future__ import annotations

import os
from typing import Tuple, Dict, Any

import numpy as np
import zarr
import torch
import torch.nn.functional as F


def _parse_Tx_Tz_from_path(zarr_path: str) -> tuple[float, float]:
    tx = np.nan
    tz = np.nan
    parts = zarr_path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("Tx_"):
            try:
                tx = float(p.split("_", 1)[1])
            except Exception:
                pass
        if p.startswith("Tz_") and p.endswith(".zarr"):
            try:
                tz = float(p.split("_", 1)[1].replace(".zarr", ""))
            except Exception:
                pass
    return float(tx), float(tz)


def _resize_hw3(field_hw3: np.ndarray, target_size=(200, 200)) -> np.ndarray:
    Ht, Wt = target_size
    if field_hw3.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3, got {field_hw3.shape}")
    H, W = field_hw3.shape[:2]
    if (H, W) == (Ht, Wt):
        return field_hw3.astype(np.float32)

    x = torch.from_numpy(field_hw3.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
    out = x.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return out


def _pick_state_idx(meta: dict, n_states: int, default: int = 0) -> int:
    s = meta.get("State", meta.get("state", None))
    try:
        s = int(s)
    except Exception:
        s = None
    if s is not None and 0 <= s < n_states:
        return s
    return int(default)


def _to_hw3(m: np.ndarray, meta: dict) -> np.ndarray:
    """
    Normalizuje różne kształty do (H,W,3).

    Obsługa typowych przypadków:
      (H,W,3)
      (K,H,W,3)         -> bierzemy 0 lub meta["State"] gdy K małe (np. 2)
      (T,H,W,3)         -> bierzemy ostatni t
      (T,K,H,W,3)       -> ostatni t i wybrany K
      (K,H,W) z dtype=(float,(3,)) albo structured -> konwersja do (K,H,W,3)
    """

    a = np.asarray(m)

    # case: dtype ma wektor (3,) w dtype zamiast osi last
    if a.ndim >= 2 and a.dtype.subdtype is not None:
        base, shp = a.dtype.subdtype
        if shp == (3,):
            a = a.view(base).reshape(*a.shape, 3)
        elif shp == (2,):  # spotykane przy 2 warstwach / 2 komponentach -> dopaduj
            tmp = a.view(base).reshape(*a.shape, 2)
            z = np.zeros((*tmp.shape[:-1], 1), dtype=tmp.dtype)
            a = np.concatenate([tmp, z], axis=-1)

    # case: structured dtype (np. pola x,y,z)
    if a.dtype.fields is not None:
        names = list(a.dtype.names)
        if len(names) >= 3:
            a = np.stack([a[names[0]], a[names[1]], a[names[2]]], axis=-1).astype(np.float32)

    if a.ndim == 3 and a.shape[-1] == 3:
        return a.astype(np.float32)

    if a.ndim == 4 and a.shape[-1] == 3:
        # (K,H,W,3) albo (T,H,W,3)
        if a.shape[0] <= 4:  # raczej "state"/warstwa
            s = _pick_state_idx(meta, a.shape[0], default=0)
            a = a[s]
        else:                # raczej czas
            a = a[-1]
        return a.astype(np.float32)

    if a.ndim == 5 and a.shape[-1] == 3:
        # (T,K,H,W,3) lub (1,K,H,W,3)
        if a.shape[1] <= 4:
            s = _pick_state_idx(meta, a.shape[1], default=0)
            a = a[-1, s]
        else:
            a = a[-1, 0]
        return a.astype(np.float32)

    # fallback: (3,H,W)
    if a.ndim == 3 and a.shape[0] == 3:
        return np.transpose(a, (1, 2, 0)).astype(np.float32)

    raise ValueError(f"Nieobsługiwany kształt m_relaxed: {a.shape}, dtype={a.dtype}")


def _read_table_last_safe(g: zarr.hierarchy.Group) -> dict:
    """
    Bezpiecznie: jeśli kolumna ma wektor na końcu (np. (2,)), nie próbujemy
    robić z tego float na siłę – zapisujemy listę albo pomijamy.
    """
    out = {}
    if "table" not in g:
        return out
    tg = g["table"]

    def _last_any(name: str):
        if name not in tg:
            return None
        try:
            arr = np.asarray(tg[name][...])
            flat = np.ravel(arr)
            if flat.size == 0:
                return None
            v = flat[-1]
            # jeśli to wektor/array -> zwróć listę
            if isinstance(v, (np.ndarray, list, tuple)):
                vv = np.asarray(v).tolist()
                return vv
            # jeśli to scalar numpy
            if hasattr(v, "item"):
                return v.item()
            return v
        except Exception:
            return None

    for k in ["ext_topologicalcharge", "mx", "my", "mz", "t", "step"]:
        v = _last_any(k)
        if v is not None:
            out[f"{k}_last"] = v
    return out


def preprocess_simulation(
    zarr_path: str,
    target_size=(200, 200),
    verbose: bool = False,
    abs_mz: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float], Dict[str, Any]]:
    g = zarr.open_group(zarr_path, mode="r")

    if "m_relaxed" not in g:
        raise KeyError(f"Brak 'm_relaxed' w {zarr_path}. Dostępne: {list(g.array_keys())}")

    meta = dict(getattr(g, "attrs", {}))
    meta.update(_read_table_last_safe(g))
    meta["source_path"] = str(zarr_path)

    m = np.asarray(g["m_relaxed"][...])
    field_hw3 = _to_hw3(m, meta)

    if abs_mz:
        field_hw3[..., 2] = np.abs(field_hw3[..., 2])

    field_hw3 = _resize_hw3(field_hw3, target_size=target_size)

    Tx, Tz = _parse_Tx_Tz_from_path(zarr_path)
    if verbose:
        print(f"[PRE] {os.path.basename(zarr_path)} raw={m.shape} -> {field_hw3.shape} | Tx={Tx} Tz={Tz}")
    return field_hw3, (Tx, Tz), meta

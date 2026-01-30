from __future__ import annotations

import os
import argparse
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from SpinGenix.postprocess.preprocess import preprocess_simulation


def _scan_zarr(sim_root: str, require_done: bool = True) -> List[str]:
    """
    Szukamy Tz_*.zarr pod Tx_*.
    Opcjonalnie wymagamy pliku *.mx3_status.done obok .zarr
    """
    out = []
    for root, dirs, files in os.walk(sim_root):
        for d in dirs:
            if d.startswith("Tz_") and d.endswith(".zarr"):
                zpath = os.path.join(root, d)
                if require_done:
                    done = os.path.join(root, d.replace(".zarr", ".mx3_status.done"))
                    if not os.path.exists(done):
                        continue
                out.append(zpath)
    return sorted(out)


def _parse_Tx_Tz(zpath: str) -> Tuple[float, float]:
    parts = zpath.replace("\\", "/").split("/")
    tx = None
    tz = None
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
    if tx is None or tz is None:
        raise ValueError(f"Nie umiem sparsować Tx/Tz z: {zpath}")
    return float(tx), float(tz)


def _stratified_pick(indices: List[int], Tx: np.ndarray, Tz: np.ndarray, n: int, bins: int, seed: int) -> List[int]:
    """
    Proste „różnorodne” próbkowanie: kwantylowe biny po Tx i Tz,
    potem round-robin po koszykach.
    """
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)

    sub_Tx = Tx[indices]
    sub_Tz = Tz[indices]

    df = pd.DataFrame({"idx": indices, "Tx": sub_Tx, "Tz": sub_Tz})

    q = min(bins, max(1, len(df)))
    df["Tx_bin"] = pd.qcut(df["Tx"], q=q, duplicates="drop")
    df["Tz_bin"] = pd.qcut(df["Tz"], q=q, duplicates="drop")

    groups = {}
    for _, r in df.iterrows():
        key = (str(r["Tx_bin"]), str(r["Tz_bin"]))
        groups.setdefault(key, []).append(int(r["idx"]))

    # shuffle w grupach
    for k in list(groups.keys()):
        rng.shuffle(groups[k])

    # round-robin wybór
    picked = []
    keys = list(groups.keys())
    rng.shuffle(keys)

    while len(picked) < n and keys:
        progress = False
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
                progress = True
        if not progress:
            break

    # jeśli wciąż brakuje -> dociągnij losowo
    if len(picked) < n:
        remaining = list(set(indices) - set(picked))
        rng.shuffle(remaining)
        picked.extend(remaining[: (n - len(picked))])

    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_root", type=str, required=True, help="np. /mnt/.../vx4")
    ap.add_argument("--meta_path", type=str, default="data/dataset/meta.h5")
    ap.add_argument("--fields_path", type=str, default="data/dataset/fields.npz")

    ap.add_argument("--n_train", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=256)

    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--require_done", action="store_true", help="wymagaj *.mx3_status.done")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.overwrite:
        if os.path.exists(args.meta_path):
            os.remove(args.meta_path)
        if os.path.exists(args.fields_path):
            os.remove(args.fields_path)

    os.makedirs(os.path.dirname(args.meta_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.fields_path), exist_ok=True)

    zarr_paths = _scan_zarr(args.sim_root, require_done=args.require_done)
    if len(zarr_paths) == 0:
        raise RuntimeError(f"Nie znaleziono Tz_*.zarr w {args.sim_root}")

    Tx = np.zeros(len(zarr_paths), dtype=float)
    Tz = np.zeros(len(zarr_paths), dtype=float)
    for i, p in enumerate(zarr_paths):
        Tx[i], Tz[i] = _parse_Tx_Tz(p)

    all_idx = list(range(len(zarr_paths)))

    # 1) stały TEST wybieramy jako pierwszy (z pełnej puli)
    test_idx = _stratified_pick(all_idx, Tx, Tz, n=args.n_test, bins=args.bins, seed=args.seed)

    # 2) TRAIN z reszty (również różnorodnie)
    remaining = list(sorted(set(all_idx) - set(test_idx)))
    train_idx = _stratified_pick(remaining, Tx, Tz, n=args.n_train, bins=args.bins, seed=args.seed + 123)

    print(f"[BUILD] Found total: {len(zarr_paths)} zarrs")
    print(f"[BUILD] Selected: train={len(train_idx)} test={len(test_idx)}")

    # Budujemy NPZ + meta
    fields_dict = {}
    meta_rows = []

    field_id = 0

    def _add_one(zpath: str, split: str):
        nonlocal field_id
        try:
            field, (tx, tz), meta = preprocess_simulation(zpath, target_size=(200, 200), verbose=args.verbose, abs_mz=True)
        except Exception as e:
            print(f"[BUILD] SKIP {zpath} ({split}): {e}")
            return

        # zapis pola
        fields_dict[str(field_id)] = field.astype(np.float32)

        # metadane: trzymamy „lekkie” i stabilne typy
        row = {
            "field_id": int(field_id),
            "split": str(split),
            "Tx_val": float(tx),
            "Tz_val": float(tz),
            "source_path": str(zpath),
        }
        # opcjonalne scalary z tabeli (jak są)
        for k in ["ext_topologicalcharge_last", "mx_last", "my_last", "mz_last", "t_last", "step_last"]:
            if k in meta:
                row[k] = float(meta[k])

        meta_rows.append(row)
        field_id += 1

    # najpierw TRAIN potem TEST (kolejność nieistotna)
    for i in train_idx:
        _add_one(zarr_paths[i], "train")
    for i in test_idx:
        _add_one(zarr_paths[i], "test")

    df = pd.DataFrame(meta_rows)

    # zapis na dysk
    np.savez_compressed(args.fields_path, **fields_dict)
    df.to_hdf(args.meta_path, key="data", mode="w")

    print(f"[BUILD] Saved meta -> {args.meta_path} (rows={len(df)})")
    print(f"[BUILD] Saved fields -> {args.fields_path} (keys={len(fields_dict)})")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
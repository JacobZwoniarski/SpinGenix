from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from SpinGenix.active_learning.samplers import sample_params
from SpinGenix.active_learning.dataset import MagnetizationDataset

try:
    from SpinGenix.simulations.swapper import SimulationManager
    from SpinGenix.postprocess.preprocess import preprocess_simulation
    HAVE_SIM = True
except Exception:
    HAVE_SIM = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--method", type=str, default="sobol", choices=["sobol", "lhs"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--Tx_min", type=float, default=10e-9)
    ap.add_argument("--Tx_max", type=float, default=100e-9)
    ap.add_argument("--Tz_min", type=float, default=10e-9)
    ap.add_argument("--Tz_max", type=float, default=100e-9)

    ap.add_argument("--Aex_min", type=float, default=None)
    ap.add_argument("--Aex_max", type=float, default=None)
    ap.add_argument("--Msat_min", type=float, default=None)
    ap.add_argument("--Msat_max", type=float, default=None)

    ap.add_argument("--out_csv", type=str, default="data/test_points.csv")

    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--sim_dir", type=str, default="/mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/")
    ap.add_argument("--prefix", type=str, default="vxTEST")

    ap.add_argument("--meta_path", type=str, default="data/dataset/meta.h5")
    ap.add_argument("--fields_path", type=str, default="data/dataset/fields.npz")

    args = ap.parse_args()

    bounds = {
        "Tx_val": (args.Tx_min, args.Tx_max),
        "Tz_val": (args.Tz_min, args.Tz_max),
    }
    if args.Aex_min is not None and args.Aex_max is not None:
        bounds["Aex"] = (args.Aex_min, args.Aex_max)
    if args.Msat_min is not None and args.Msat_max is not None:
        bounds["Msat"] = (args.Msat_min, args.Msat_max)

    samples = sample_params(bounds, n=args.n, method=args.method, seed=args.seed)
    df = pd.DataFrame(samples)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[TESTSET] Saved points to {args.out_csv} ({len(df)} rows)")

    if (args.submit or args.preprocess) and not HAVE_SIM:
        raise RuntimeError("SimulationManager/preprocess_simulation not available in this environment.")

    if args.submit:
        params = {
            "Tx": df["Tx_val"].to_numpy(),
            "Tz": df["Tz_val"].to_numpy(),
        }
        if "Aex" in df.columns:
            params["Aex"] = df["Aex"].to_numpy()
        if "Msat" in df.columns:
            params["Msat"] = df["Msat"].to_numpy()

        sim = SimulationManager(
            main_path=args.sim_dir,
            destination_path=args.sim_dir,
            prefix=args.prefix
        )
        print("[TESTSET] Submitting simulations...")
        sim.submit_all_simulations(params, last_param_name="Tz", pairs=True, sbatch=True)

    if args.preprocess:
        # <<< FIX: prefix w ścieżkach >>>
        expected_paths = [
            os.path.join(args.sim_dir, args.prefix, f"Tx_{tx}", f"Tz_{tz}.zarr")
            for tx, tz in zip(df["Tx_val"].to_numpy(), df["Tz_val"].to_numpy())
        ]

        param_cols = ["Tx_val", "Tz_val"]
        for extra in ["Aex", "Msat"]:
            if extra in df.columns:
                param_cols.append(extra)

        ds_full = MagnetizationDataset(
            meta_path=args.meta_path,
            fields_path=args.fields_path,
            split=None,
            param_cols=param_cols,
            device="cpu",
            target_size=(200, 200),
        )

        print("[TESTSET] Preprocessing + adding to dataset as split=test ...")
        added = 0
        for p in expected_paths:
            if not os.path.exists(p):
                continue

            field, (Tx, Tz), meta = preprocess_simulation(p)

            md = dict(meta)
            md["Tx_val"] = float(Tx)
            md["Tz_val"] = float(Tz)

            ds_full.add_sample(field, md, split="test")
            added += 1

        ds_full.save(args.meta_path, args.fields_path)
        print(f"[TESTSET] Done. Added {added}/{len(expected_paths)} test samples (split=test) and saved dataset.")


if __name__ == "__main__":
    main()
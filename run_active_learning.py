#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-spingenix"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import torch  # noqa: E402

from active_learning.loop import ActiveLearningLoop  # noqa: E402


def nm_range(min_nm, max_nm):
    return (float(min_nm) * 1e-9, float(max_nm) * 1e-9)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the SpinGenix active-learning loop."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Train/acquire and write acquisition CSV, but do not submit Slurm jobs or mutate the dataset.",
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="Submit selected points to Slurm, wait for completed zarr outputs, and append them to the dataset.",
    )

    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--grid-points", type=int, default=40)
    parser.add_argument("--k-new", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--tx-min-nm", type=float, default=10.0)
    parser.add_argument("--tx-max-nm", type=float, default=110.0)
    parser.add_argument("--tz-min-nm", type=float, default=10.0)
    parser.add_argument("--tz-max-nm", type=float, default=110.0)
    parser.add_argument("--acquisition-min-distance-nm", type=float, default=None)

    parser.add_argument("--meta-path", default="data/dataset/meta.h5")
    parser.add_argument("--fields-path", default="data/dataset/fields.npz")
    parser.add_argument("--dataset-dir", default="data/dataset")
    parser.add_argument("--normalizer-path", default="data/dataset/param_normalizer.json")
    parser.add_argument("--registry-dir", default="data/registry")
    parser.add_argument("--results-dir", default="results/active_learning")
    parser.add_argument(
        "--simulations-dir",
        default="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/",
    )
    parser.add_argument("--simulation-prefix", default="vxAL")

    parser.add_argument("--poll-interval", type=int, default=300)
    parser.add_argument("--max-wait-hours", type=float, default=24.0)
    parser.add_argument("--amumax-bin", default=None)
    parser.add_argument("--cuda-module", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    acquisition_min_distance = None
    if args.acquisition_min_distance_nm is not None:
        acquisition_min_distance = float(args.acquisition_min_distance_nm) * 1e-9

    print(
        "[run_active_learning] "
        f"mode={'submit' if args.submit else 'dry-run'}, "
        f"device={device}, iterations={args.iterations}, epochs={args.epochs}, "
        f"grid={args.grid_points}x{args.grid_points}, k_new={args.k_new}",
        flush=True,
    )

    al = ActiveLearningLoop(
        meta_path=args.meta_path,
        fields_path=args.fields_path,
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        registry_path=args.registry_dir,
        normalizer_path=args.normalizer_path,
        simulations_dir=args.simulations_dir,
        Tx_range=nm_range(args.tx_min_nm, args.tx_max_nm),
        Tz_range=nm_range(args.tz_min_nm, args.tz_max_nm),
        grid_points=args.grid_points,
        k_new=args.k_new,
        mc_samples=args.mc_samples,
        acquisition_min_distance=acquisition_min_distance,
        poll_interval=args.poll_interval,
        max_wait_hours=args.max_wait_hours,
        submit_simulations=args.submit,
        simulation_prefix=args.simulation_prefix,
        amumax_bin=args.amumax_bin,
        cuda_module=args.cuda_module,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    al.run(iterations=args.iterations)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the local V1 pipeline after vx5 simulations are complete enough.

This script does not submit Slurm jobs. It consumes completed .zarr outputs and
orchestrates:
  status report -> dataset build -> smoke checks -> optional params->field training.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n$ " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True, cwd=ROOT)


def count_complete(status_csv):
    with open(status_csv, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    complete = sum(1 for row in rows if row["status"] == "complete")
    return complete, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="vx5")
    parser.add_argument("--raw-root", default="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations")
    parser.add_argument("--dataset-dir", default="data/dataset")
    parser.add_argument("--registry-dir", default="data/registry")
    parser.add_argument("--results-dir", default="results/v1_pipeline")
    parser.add_argument("--source", default="initial_lhs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-complete", type=int, default=1)
    parser.add_argument(
        "--require-all-complete",
        action="store_true",
        help="Fail if any discovered point is not complete.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-train-epochs", type=int, default=0)
    parser.add_argument("--train-param-surrogate", action="store_true")
    parser.add_argument("--surrogate-epochs", type=int, default=20)
    parser.add_argument("--surrogate-batch-size", type=int, default=8)
    parser.add_argument("--sample-grid", action="store_true")
    parser.add_argument("--grid-points", type=int, default=40)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    status_csv = results_dir / "status" / f"{args.prefix}_status.csv"
    dataset_dir = Path(args.dataset_dir)
    meta_path = dataset_dir / "meta.h5"
    fields_path = dataset_dir / "fields.npz"
    normalizer_path = dataset_dir / "param_normalizer.json"

    run([
        sys.executable,
        "scripts/report_simulation_status.py",
        "--raw-root",
        args.raw_root,
        "--prefix",
        args.prefix,
        "--csv",
        status_csv,
    ])
    complete, total = count_complete(status_csv)
    print(f"complete simulations: {complete}/{total}")

    if complete < args.min_complete:
        raise SystemExit(
            f"Only {complete} complete simulations; need at least {args.min_complete}."
        )
    if args.require_all_complete and complete != total:
        raise SystemExit(f"Only {complete}/{total} simulations are complete.")

    run([
        sys.executable,
        "scripts/build_dataset_from_zarr.py",
        "--raw-root",
        args.raw_root,
        "--prefix",
        args.prefix,
        "--out-dir",
        dataset_dir,
        "--registry-dir",
        args.registry_dir,
        "--split",
        args.split,
        "--source",
        args.source,
    ])

    smoke_cmd = [
        sys.executable,
        "scripts/smoke_v1_pipeline.py",
        "--raw-root",
        args.raw_root,
        "--prefix",
        args.prefix,
        "--meta-path",
        meta_path,
        "--fields-path",
        fields_path,
        "--normalizer-path",
        normalizer_path,
        "--out-dir",
        results_dir / "smoke",
        "--device",
        args.device,
        "--train-epochs",
        args.smoke_train_epochs,
    ]
    run(smoke_cmd)

    checkpoint_path = results_dir / "param_surrogate" / "param_surrogate.pt"
    if args.train_param_surrogate:
        run([
            sys.executable,
            "scripts/train_param_surrogate.py",
            "--meta-path",
            meta_path,
            "--fields-path",
            fields_path,
            "--normalizer-path",
            normalizer_path,
            "--out-dir",
            results_dir / "param_surrogate",
            "--device",
            args.device,
            "--epochs",
            args.surrogate_epochs,
            "--batch-size",
            args.surrogate_batch_size,
        ])

    if args.sample_grid:
        if not checkpoint_path.exists():
            raise SystemExit(
                f"Missing checkpoint for grid sampling: {checkpoint_path}. "
                "Use --train-param-surrogate first."
            )
        run([
            sys.executable,
            "scripts/sample_param_surrogate_grid.py",
            "--checkpoint",
            checkpoint_path,
            "--out-dir",
            results_dir / "param_surrogate_grid",
            "--device",
            args.device,
            "--grid-points",
            args.grid_points,
        ])

    print("\nV1 pipeline completed.")


if __name__ == "__main__":
    main()

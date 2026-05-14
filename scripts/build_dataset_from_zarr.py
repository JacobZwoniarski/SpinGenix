#!/usr/bin/env python3
"""
Build SpinGenix training files from existing raw .zarr simulations.

Output format:
  data/dataset/meta.h5     - metadata table under key "data"
  data/dataset/fields.npz  - arrays keyed as "0", "1", ...
"""

import argparse
import getpass
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

REQUIRED_METADATA_COLUMNS = {
    "Tx_val",
    "Tz_val",
    "MeanMz_signed",
    "MeanMz_abs",
    "MeanMz",
    "MeanMx",
    "MeanMy",
    "Q",
    "State",
    "Aex",
    "Msat",
    "source_path",
}


def parse_named_param(path, name):
    pattern = re.compile(rf"{name}_?(\d+(?:\.\d+)?e[+-]?\d+)", re.IGNORECASE)
    for part in reversed(Path(path).parts):
        match = pattern.search(part)
        if match:
            return float(match.group(1))
    return None


def discover_zarr(root, prefixes=None):
    root = Path(root)
    prefixes = set(prefixes or [])

    for path in root.rglob("*.zarr"):
        if prefixes:
            try:
                rel_parts = path.relative_to(root).parts
            except ValueError:
                rel_parts = path.parts
            if not rel_parts or rel_parts[0] not in prefixes:
                continue

        if not (path / "m_relaxed" / ".zarray").exists():
            continue
        if not (path / ".zattrs").exists():
            continue

        Tx = parse_named_param(path, "Tx")
        Tz = parse_named_param(path, "Tz")
        if Tx is None or Tz is None:
            continue

        yield {
            "path": path,
            "Tx_val": Tx,
            "Tz_val": Tz,
        }


def select_stratified(records, bins, per_bin, seed):
    if per_bin is None:
        return records
    if not records:
        return []

    rng = random.Random(seed)
    tx = np.array([r["Tx_val"] for r in records], dtype=float)
    tz = np.array([r["Tz_val"] for r in records], dtype=float)

    tx_edges = np.linspace(tx.min(), tx.max(), bins + 1)
    tz_edges = np.linspace(tz.min(), tz.max(), bins + 1)

    buckets = {}
    for record in records:
        i = np.searchsorted(tx_edges, record["Tx_val"], side="right") - 1
        j = np.searchsorted(tz_edges, record["Tz_val"], side="right") - 1
        i = min(max(i, 0), bins - 1)
        j = min(max(j, 0), bins - 1)
        buckets.setdefault((i, j), []).append(record)

    selected = []
    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)
        selected.extend(bucket_records[:per_bin])

    selected.sort(key=lambda r: (r["Tx_val"], r["Tz_val"], str(r["path"])))
    return selected


def validate_dataset(metadata_rows, fields, target_size=(200, 200)):
    missing = REQUIRED_METADATA_COLUMNS.difference(metadata_rows[0].keys())
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    expected_shape = (*target_size, 3)
    bad_shapes = {
        key: value.shape
        for key, value in fields.items()
        if value.shape != expected_shape
    }
    if bad_shapes:
        preview = list(bad_shapes.items())[:5]
        raise ValueError(f"Fields with invalid shape; expected {expected_shape}: {preview}")

    if len(metadata_rows) != len(fields):
        raise ValueError(
            f"Metadata length {len(metadata_rows)} != fields length {len(fields)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations",
        help="Directory containing simulation prefix folders such as vx5/v7/vxAL.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/dataset",
        help="Directory where meta.h5 and fields.npz will be written.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="Limit to one simulation prefix. May be repeated, e.g. --prefix v7 --prefix vx5.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Number of Tx/Tz bins per axis for stratified seed selection.",
    )
    parser.add_argument(
        "--per-bin",
        type=int,
        default=None,
        help="If set, keep at most this many samples per Tx/Tz bin. Omit to use all records.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap after stratified selection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    records = list(discover_zarr(args.raw_root, prefixes=args.prefixes))
    records.sort(key=lambda r: (r["Tx_val"], r["Tz_val"], str(r["path"])))

    selected = select_stratified(records, args.bins, args.per_bin, args.seed)
    if args.max_samples is not None:
        random.Random(args.seed).shuffle(selected)
        selected = sorted(
            selected[: args.max_samples],
            key=lambda r: (r["Tx_val"], r["Tz_val"], str(r["path"])),
        )

    print(f"Discovered {len(records)} candidate .zarr simulations.")
    print(f"Selected {len(selected)} simulations for dataset build.")

    if args.dry_run:
        for record in selected[:20]:
            print(record["path"])
        if len(selected) > 20:
            print(f"... {len(selected) - 20} more")
        return

    import pandas as pd
    from postprocess.preprocess import preprocess_simulation

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = {}
    metadata_rows = []
    failures = []

    for idx, record in enumerate(selected):
        zarr_path = record["path"]
        try:
            field, (Tx, Tz), metadata = preprocess_simulation(
                str(zarr_path),
                target_size=(200, 200),
                verbose=args.verbose,
            )
            fields[str(len(metadata_rows))] = field
            metadata_rows.append({
                "Tx_val": Tx,
                "Tz_val": Tz,
                **metadata,
            })
            print(f"[{idx + 1}/{len(selected)}] OK {zarr_path}")
        except Exception as exc:
            failures.append({
                "path": os.path.abspath(zarr_path),
                "error": repr(exc),
            })
            print(f"[{idx + 1}/{len(selected)}] FAIL {zarr_path}: {exc}")

    if not metadata_rows:
        raise RuntimeError("No simulations were successfully preprocessed.")

    validate_dataset(metadata_rows, fields, target_size=(200, 200))

    meta_path = out_dir / "meta.h5"
    fields_path = out_dir / "fields.npz"

    pd.DataFrame(metadata_rows).to_hdf(meta_path, key="data", mode="w", format="table")
    np.savez_compressed(fields_path, **fields)

    print(f"Saved metadata: {meta_path}")
    print(f"Saved fields:   {fields_path}")
    print(f"Samples:        {len(metadata_rows)}")

    if failures:
        failures_path = out_dir / "preprocess_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Failures:       {len(failures)} written to {failures_path}")


if __name__ == "__main__":
    main()

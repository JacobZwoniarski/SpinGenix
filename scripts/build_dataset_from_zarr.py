#!/usr/bin/env python3
"""
Build SpinGenix training files from existing raw .zarr simulations.

Output format:
  data/dataset/meta.h5     - metadata table under key "data"
  data/dataset/fields.npz  - arrays keyed as "0", "1", ...
"""

import argparse
import getpass
import json
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

from active_learning.registry import (
    PARAM_COLUMNS_V1,
    SPLITS,
    compute_param_hash,
    infer_prefix_from_path,
    make_registry_row,
    simulation_id,
    upsert_registry,
)

REQUIRED_METADATA_COLUMNS = {
    "simulation_id",
    "param_hash",
    "split",
    "field_key",
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
    "Ty_val",
    "Nx",
    "Ny",
    "Nz",
    "dx",
    "dy",
    "dz",
    "target_Nx",
    "target_Ny",
    "target_Nz",
    "source_path",
}

ZATTR_COLUMNS = {
    "Ty": "Ty_val",
    "Nx": "Nx",
    "Ny": "Ny",
    "Nz": "Nz",
    "dx": "dx",
    "dy": "dy",
    "dz": "dz",
    "Aex": "Aex",
    "Msat": "Msat",
}


def parse_named_param(path, name):
    pattern = re.compile(rf"{name}_?(\d+(?:\.\d+)?e[+-]?\d+)", re.IGNORECASE)
    for part in reversed(Path(path).parts):
        match = pattern.search(part)
        if match:
            return float(match.group(1))
    return None


def read_zattrs(zarr_path):
    try:
        with open(Path(zarr_path) / ".zattrs", "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def zattrs_record_fields(attrs):
    fields = {}
    for attr_key, column in ZATTR_COLUMNS.items():
        if attr_key not in attrs:
            continue
        if column in {"Nx", "Ny", "Nz"}:
            value = as_int(attrs[attr_key])
        else:
            value = as_float(attrs[attr_key])
        if value is not None:
            fields[column] = value
    return fields


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

        attrs = read_zattrs(path)
        Tx = as_float(attrs.get("Tx"), parse_named_param(path, "Tx"))
        Tz = as_float(attrs.get("Tz"), parse_named_param(path, "Tz"))
        if Tx is None or Tz is None:
            continue

        record = {
            "path": path,
            "Tx_val": Tx,
            "Tz_val": Tz,
        }
        record.update(zattrs_record_fields(attrs))
        yield record


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


def registry_params(record, param_columns):
    return {column: record[column] for column in param_columns if column in record}


def registry_extra(record, metadata=None):
    extra = {}
    metadata = metadata or {}
    for column in [
        "Ty_val",
        "Aex",
        "Msat",
        "Nx",
        "Ny",
        "Nz",
        "dx",
        "dy",
        "dz",
        "target_Nx",
        "target_Ny",
        "target_Nz",
    ]:
        if column in metadata and metadata[column] is not None:
            extra[column] = metadata[column]
        elif column in record and record[column] is not None:
            extra[column] = record[column]
    return extra


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
    parser.add_argument(
        "--registry-dir",
        default="data/registry",
        help="Directory for simulations.csv/parquet registry updates.",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Build dataset artifacts without updating the central simulation registry.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="train",
        help="Split label assigned to all selected simulations in this build.",
    )
    parser.add_argument(
        "--source",
        default="initial_lhs",
        help="Registry source label, e.g. initial_lhs, active_learning, manual.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Active-learning iteration number for registry rows.",
    )
    parser.add_argument(
        "--param-column",
        action="append",
        dest="param_columns",
        default=None,
        help="Physical parameter column used for param_hash. Repeat for N-D runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    param_columns = tuple(args.param_columns or PARAM_COLUMNS_V1)

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
    print(f"split/source: {args.split}/{args.source}")
    print(f"param columns: {', '.join(param_columns)}")

    if args.dry_run:
        for record in selected[:20]:
            params = registry_params(record, param_columns)
            phash = compute_param_hash(params, param_names=param_columns)
            prefix = infer_prefix_from_path(record["path"], raw_root=args.raw_root)
            print(f"{record['path']}  hash={phash[:12]}  id={simulation_id(prefix, phash)}")
        if len(selected) > 20:
            print(f"... {len(selected) - 20} more")
        return

    import pandas as pd
    from postprocess.preprocess import preprocess_simulation

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = {}
    metadata_rows = []
    registry_rows = []
    failures = []

    for idx, record in enumerate(selected):
        zarr_path = record["path"]
        prefix = infer_prefix_from_path(zarr_path, raw_root=args.raw_root)
        params = registry_params(record, param_columns)
        phash = compute_param_hash(params, param_names=param_columns)
        sim_id = simulation_id(prefix, phash)
        try:
            field, (Tx, Tz), metadata = preprocess_simulation(
                str(zarr_path),
                target_size=(200, 200),
                verbose=args.verbose,
            )
            params = {**params, "Tx_val": Tx, "Tz_val": Tz}
            phash = compute_param_hash(params, param_names=param_columns)
            sim_id = simulation_id(prefix, phash)
            field_key = str(len(metadata_rows))
            fields[field_key] = field
            metadata_rows.append({
                "simulation_id": sim_id,
                "param_hash": phash,
                "split": args.split,
                "field_key": field_key,
                "Tx_val": Tx,
                "Tz_val": Tz,
                **metadata,
            })
            registry_rows.append(make_registry_row(
                params={**params, "Tx_val": Tx, "Tz_val": Tz},
                prefix=prefix,
                split=args.split,
                source=args.source,
                status="done",
                iteration=args.iteration,
                param_names=param_columns,
                zarr_path=zarr_path,
                field_path=str(Path(args.out_dir) / "fields.npz"),
                field_key=field_key,
                extra=registry_extra(record, metadata),
            ))
            print(f"[{idx + 1}/{len(selected)}] OK {zarr_path}")
        except Exception as exc:
            registry_rows.append(make_registry_row(
                params=params,
                prefix=prefix,
                split=args.split,
                source=args.source,
                status="preprocess_failed",
                iteration=args.iteration,
                param_names=param_columns,
                zarr_path=zarr_path,
                error_message=repr(exc),
                extra=registry_extra(record),
            ))
            failures.append({
                "path": os.path.abspath(zarr_path),
                "simulation_id": sim_id,
                "param_hash": phash,
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

    if not args.no_registry:
        _, registry_paths = upsert_registry(registry_rows, registry_dir=args.registry_dir)
        print(f"Registry CSV:   {registry_paths['csv']}")
        if registry_paths["parquet"] is not None:
            print(f"Registry PQ:    {registry_paths['parquet']}")
        elif registry_paths["parquet_error"]:
            print(f"Registry PQ:    skipped ({registry_paths['parquet_error']})")

    if failures:
        failures_path = out_dir / "preprocess_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Failures:       {len(failures)} written to {failures_path}")


if __name__ == "__main__":
    main()

"""
Central registry helpers for SpinGenix simulations.

The registry is intentionally independent of model code: it records which
physical parameter points exist, which split they belong to, and where their
raw/processed artifacts live. Active learning should treat every registry row
as an exclusion candidate unless a workflow explicitly says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


PARAM_COLUMNS_V1 = ("Tx_val", "Tz_val")

TRAINING_SPLITS = {"train", "val"}
HOLDOUT_SPLITS = {"test_holdout", "boundary_holdout", "ood_holdout"}
SPLITS = tuple(sorted(TRAINING_SPLITS | HOLDOUT_SPLITS))


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_param_value(value, precision=12):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.{precision}e}"


def canonical_params(params, param_names=None, precision=12):
    """
    Return a stable list of parameter name/value pairs for hashing.

    Values are formatted as scientific-notation strings to make hashes stable
    across JSON, pandas, and NumPy float round trips.
    """
    if param_names is None:
        param_names = sorted(params.keys())

    items = []
    for name in param_names:
        if name not in params:
            continue
        items.append({
            "name": str(name),
            "value": _format_param_value(params[name], precision=precision),
        })
    return items


def compute_param_hash(params, param_names=None, precision=12):
    payload = canonical_params(params, param_names=param_names, precision=precision)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def simulation_id(prefix, param_hash):
    prefix = str(prefix or "unknown").strip().replace("/", "_")
    return f"{prefix}_{param_hash[:16]}"


def infer_prefix_from_path(path, raw_root=None):
    path = Path(path)
    if raw_root is not None:
        try:
            rel = path.relative_to(Path(raw_root))
            return rel.parts[0] if rel.parts else None
        except ValueError:
            pass

    parts = path.parts
    for idx, part in enumerate(parts):
        if part.startswith("Tx_") and idx > 0:
            return parts[idx - 1]
    return None


def make_registry_row(
    *,
    params,
    prefix,
    split,
    source,
    status,
    iteration=0,
    param_names=PARAM_COLUMNS_V1,
    zarr_path=None,
    field_path=None,
    field_key=None,
    error_message=None,
    created_at=None,
    updated_at=None,
    extra=None,
):
    phash = compute_param_hash(params, param_names=param_names)
    row = {
        "simulation_id": simulation_id(prefix, phash),
        "param_hash": phash,
        "prefix": prefix,
        "iteration": int(iteration),
        "source": source,
        "status": status,
        "split": split,
        "zarr_path": os.path.abspath(zarr_path) if zarr_path else "",
        "field_path": os.path.abspath(field_path) if field_path else "",
        "field_key": "" if field_key is None else str(field_key),
        "created_at": created_at or utc_now_iso(),
        "updated_at": updated_at or utc_now_iso(),
        "error_message": error_message or "",
    }
    for name in param_names:
        if name in params:
            row[name] = params[name]
    if extra:
        row.update(extra)
    return row


def registry_file_candidates(registry_path):
    path = Path(registry_path)
    if path.suffix:
        return [path]
    return [path / "simulations.parquet", path / "simulations.csv"]


def read_registry(registry_path="data/registry"):
    import pandas as pd

    for path in registry_file_candidates(registry_path):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported registry format: {path}")
    raise FileNotFoundError(f"No registry file found under {registry_path}")


def validate_strict_holdout(registry_df):
    if registry_df.empty or "param_hash" not in registry_df or "split" not in registry_df:
        return

    leaks = []
    grouped = registry_df.dropna(subset=["param_hash"]).groupby("param_hash")
    for phash, group in grouped:
        splits = {str(split) for split in group["split"].dropna()}
        if splits & TRAINING_SPLITS and splits & HOLDOUT_SPLITS:
            leaks.append((phash, sorted(splits)))

    if leaks:
        preview = ", ".join(f"{phash[:12]}:{splits}" for phash, splits in leaks[:5])
        raise ValueError(
            "Strict holdout violation: identical param_hash appears in training "
            f"and holdout splits ({preview})"
        )


def upsert_registry(new_rows, registry_dir="data/registry"):
    import pandas as pd

    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        raise ValueError("No registry rows to write.")

    try:
        existing_df = read_registry(registry_dir)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    validate_strict_holdout(combined)

    if "simulation_id" in combined:
        combined = combined.drop_duplicates(subset=["simulation_id"], keep="last")
    combined = combined.reset_index(drop=True)

    csv_path = registry_dir / "simulations.csv"
    parquet_path = registry_dir / "simulations.parquet"

    combined.to_csv(csv_path, index=False)

    parquet_error = None
    try:
        combined.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pandas needs pyarrow or fastparquet.
        parquet_error = f"{type(exc).__name__}: {exc}"

    return combined, {
        "csv": csv_path,
        "parquet": parquet_path if parquet_path.exists() else None,
        "parquet_error": parquet_error,
    }


def registry_points_and_hashes(registry_path="data/registry", param_columns=PARAM_COLUMNS_V1):
    try:
        df = read_registry(registry_path)
    except FileNotFoundError:
        return None, set()

    points = None
    if all(column in df.columns for column in param_columns):
        points = df.loc[:, list(param_columns)].dropna().to_numpy(dtype=float)

    hashes = set()
    if "param_hash" in df.columns:
        hashes = {str(value) for value in df["param_hash"].dropna()}

    return points, hashes

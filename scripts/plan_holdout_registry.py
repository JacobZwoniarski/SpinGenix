#!/usr/bin/env python3
"""
Reserve strict-holdout parameter points before active learning.

This script does not submit simulations. It writes planned holdout rows to the
central registry so acquisition can exclude them from future training picks.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.registry import (  # noqa: E402
    HOLDOUT_SPLITS,
    PARAM_COLUMNS_V1,
    compute_param_hash,
    make_registry_row,
    registry_points_and_hashes,
    upsert_registry,
)


def nm_to_si(value):
    return float(value) * 1e-9


def si_to_nm(value):
    return float(value) * 1e9


def generate_lhs_points(count, tx_range, tz_range, seed):
    rng = random.Random(seed)
    tx_slots = list(range(count))
    tz_slots = list(range(count))
    rng.shuffle(tx_slots)
    rng.shuffle(tz_slots)

    points = []
    for i in range(count):
        tx_unit = (tx_slots[i] + rng.random()) / count
        tz_unit = (tz_slots[i] + rng.random()) / count
        tx = tx_range[0] + tx_unit * (tx_range[1] - tx_range[0])
        tz = tz_range[0] + tz_unit * (tz_range[1] - tz_range[0])
        points.append((tx, tz))
    return points


def too_close(point, points, min_distance):
    if points is None or min_distance <= 0:
        return False
    point = np.asarray(point, dtype=float)
    for other in points:
        if np.linalg.norm(point - np.asarray(other, dtype=float)) < min_distance:
            return True
    return False


def select_holdout_points(args, param_columns):
    tx_range = (nm_to_si(args.tx_min_nm), nm_to_si(args.tx_max_nm))
    tz_range = (nm_to_si(args.tz_min_nm), nm_to_si(args.tz_max_nm))
    min_distance = nm_to_si(args.min_distance_nm)

    existing_points, existing_hashes = registry_points_and_hashes(
        args.registry_dir,
        param_columns=param_columns,
    )
    if existing_points is None:
        existing_points = np.empty((0, 2), dtype=float)

    selected = []
    selected_hashes = set()
    attempts = max(args.count * 20, args.count + 100)
    candidates = generate_lhs_points(attempts, tx_range, tz_range, args.seed)

    for tx, tz in candidates:
        params = {"Tx_val": tx, "Tz_val": tz}
        phash = compute_param_hash(params, param_names=param_columns)
        if phash in existing_hashes or phash in selected_hashes:
            continue
        if too_close((tx, tz), selected, min_distance):
            continue
        if too_close((tx, tz), existing_points, min_distance):
            continue

        selected.append((tx, tz))
        selected_hashes.add(phash)
        if len(selected) >= args.count:
            break

    if len(selected) < args.count:
        raise RuntimeError(
            f"Could only reserve {len(selected)}/{args.count} holdout points. "
            "Reduce --min-distance-nm or widen the parameter range."
        )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", default="data/registry")
    parser.add_argument("--prefix", default="vx_holdout")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tx-min-nm", type=float, default=10.0)
    parser.add_argument("--tx-max-nm", type=float, default=100.0)
    parser.add_argument("--tz-min-nm", type=float, default=10.0)
    parser.add_argument("--tz-max-nm", type=float, default=100.0)
    parser.add_argument(
        "--min-distance-nm",
        type=float,
        default=0.0,
        help="Optional Euclidean exclusion radius in Tx/Tz nanometres.",
    )
    parser.add_argument("--split", choices=sorted(HOLDOUT_SPLITS), default="test_holdout")
    parser.add_argument("--source", default="strict_holdout_lhs")
    parser.add_argument(
        "--param-column",
        action="append",
        dest="param_columns",
        default=None,
        help="Physical parameter column used for param_hash. Repeat for N-D runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    param_columns = tuple(args.param_columns or PARAM_COLUMNS_V1)
    points = select_holdout_points(args, param_columns)
    rows = [
        make_registry_row(
            params={"Tx_val": tx, "Tz_val": tz},
            prefix=args.prefix,
            split=args.split,
            source=args.source,
            status="planned",
            iteration=0,
            param_names=param_columns,
        )
        for tx, tz in points
    ]

    print(f"planned holdout rows: {len(rows)}")
    print(f"split/source: {args.split}/{args.source}")
    print("first points [nm]:")
    for tx, tz in points[:10]:
        print(f"  Tx={si_to_nm(tx):8.3f}, Tz={si_to_nm(tz):8.3f}")

    if args.dry_run:
        print("dry run only; omit --dry-run to write registry rows")
        return

    _, paths = upsert_registry(rows, registry_dir=args.registry_dir)
    print(f"Registry CSV: {paths['csv']}")
    if paths["parquet"] is not None:
        print(f"Registry PQ:  {paths['parquet']}")
    elif paths["parquet_error"]:
        print(f"Registry PQ:  skipped ({paths['parquet_error']})")


if __name__ == "__main__":
    main()

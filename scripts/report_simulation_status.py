#!/usr/bin/env python3
"""
Report raw simulation status for a SpinGenix prefix such as vx5 or vxAL.

The report is intentionally filesystem-based, so it works even when Slurm is
temporarily unreachable from the current shell/session.
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def parse_named_param(path, name):
    pattern = re.compile(rf"{name}_?(\d+(?:\.\d+)?e[+-]?\d+)", re.IGNORECASE)
    for part in Path(path).parts:
        match = pattern.search(part)
        if match:
            return float(match.group(1))
    return None


def format_param(value):
    return format(float(value), ".5g")


def discover_points(prefix_dir):
    points = {}
    candidates = []
    candidates.extend(prefix_dir.rglob("*.zarr"))
    candidates.extend(prefix_dir.rglob("*.mx3"))
    candidates.extend(prefix_dir.rglob("*.mx3_status.*"))

    for path in candidates:
        tx = parse_named_param(path, "Tx")
        tz = parse_named_param(path, "Tz")
        if tx is None or tz is None:
            continue
        points[(format_param(tx), format_param(tz))] = (tx, tz)
    return [points[key] for key in sorted(points)]


def point_dir(prefix_dir, tx):
    return prefix_dir / f"Tx_{format_param(tx)}"


def zarr_path(prefix_dir, tx, tz):
    return point_dir(prefix_dir, tx) / f"Tz_{format_param(tz)}.zarr"


def status_file(prefix_dir, tx, tz, status):
    pattern = f"Tz_{format_param(tz)}.mx3_status.{status}*"
    matches = sorted(point_dir(prefix_dir, tx).glob(pattern))
    return matches[-1] if matches else None


def is_complete_zarr(path):
    return (
        path.exists()
        and (path / ".zattrs").exists()
        and (path / "m_relaxed" / ".zarray").exists()
    )


def classify(prefix_dir, tx, tz):
    zarr = zarr_path(prefix_dir, tx, tz)
    complete = is_complete_zarr(zarr)
    done = status_file(prefix_dir, tx, tz, "done")
    lock = status_file(prefix_dir, tx, tz, "lock")
    interrupted = status_file(prefix_dir, tx, tz, "interrupted")
    mx3 = point_dir(prefix_dir, tx) / f"Tz_{format_param(tz)}.mx3"

    if complete:
        status = "complete"
    elif lock:
        status = "locked"
    elif done:
        status = "done_status_incomplete_zarr"
    elif interrupted:
        status = "interrupted"
    elif zarr.exists():
        status = "incomplete_zarr"
    elif mx3.exists():
        status = "prepared_mx3"
    else:
        status = "missing"

    return {
        "Tx_val": tx,
        "Tz_val": tz,
        "Tx_nm": tx * 1e9,
        "Tz_nm": tz * 1e9,
        "status": status,
        "complete_zarr": int(complete),
        "zarr_path": str(zarr),
        "status_file": str(lock or done or interrupted or ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations")
    parser.add_argument("--prefix", default="vx5")
    parser.add_argument("--csv", default=None, help="Optional path to write a row-level CSV report.")
    parser.add_argument("--limit", type=int, default=20, help="How many non-complete rows to preview.")
    args = parser.parse_args()

    prefix_dir = Path(args.raw_root) / args.prefix
    if not prefix_dir.exists():
        raise FileNotFoundError(f"Prefix directory does not exist: {prefix_dir}")

    rows = [classify(prefix_dir, tx, tz) for tx, tz in discover_points(prefix_dir)]
    rows.sort(key=lambda row: (row["status"], row["Tx_val"], row["Tz_val"]))
    counts = Counter(row["status"] for row in rows)

    print(f"prefix: {args.prefix}")
    print(f"points: {len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    incomplete = [row for row in rows if row["status"] != "complete"]
    if incomplete:
        print("first non-complete points [nm]:")
        for row in incomplete[: args.limit]:
            print(
                f"  {row['status']:28s} "
                f"Tx={row['Tx_nm']:8.3f}, Tz={row['Tz_nm']:8.3f} "
                f"{row['status_file']}"
            )

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()

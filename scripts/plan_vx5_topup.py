#!/usr/bin/env python3
"""
Plan or submit a small balanced vx5 top-up batch.

This is intentionally narrower than submit_vx5_initial.py. It is meant for the
bootstrap-repair phase where existing vx5 points under-cover high Tx values and
we want a small number of additional simulations, not a full dense map.
"""

import argparse
import getpass
import random
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMULATIONS_DIR = ROOT / "simulations"


def nm(value):
    return float(value) * 1e9


def format_param(value):
    return format(value, ".5g")


def parse_named_param(path, name):
    pattern = re.compile(rf"{name}_?(\d+(?:\.\d+)?e[+-]?\d+)", re.IGNORECASE)
    for part in Path(path).parts:
        match = pattern.search(part)
        if match:
            return float(match.group(1))
    return None


def is_complete_zarr(path):
    path = Path(path)
    return (path / ".zattrs").exists() and (path / "m_relaxed" / ".zarray").exists()


def discover_points(raw_root, prefix):
    prefix_dir = Path(raw_root) / prefix
    points = {}
    if not prefix_dir.exists():
        return []

    candidates = []
    candidates.extend(prefix_dir.rglob("*.zarr"))
    candidates.extend(prefix_dir.rglob("*.mx3"))
    candidates.extend(prefix_dir.rglob("*.mx3_status.*"))

    for path in candidates:
        tx = parse_named_param(path, "Tx")
        tz = parse_named_param(path, "Tz")
        if tx is None or tz is None:
            continue
        key = (format_param(tx), format_param(tz))
        record = points.setdefault(
            key,
            {
                "Tx": tx,
                "Tz": tz,
                "complete": False,
                "locked": False,
                "interrupted": False,
                "prepared": False,
            },
        )
        text = str(path)
        if text.endswith(".zarr"):
            record["complete"] = is_complete_zarr(path)
        elif text.endswith(".mx3"):
            record["prepared"] = True
        elif ".mx3_status.lock" in text:
            record["locked"] = True
        elif ".mx3_status.interrupted" in text:
            record["interrupted"] = True

    return sorted(points.values(), key=lambda r: (r["Tx"], r["Tz"]))


def status(record):
    if record["complete"]:
        return "complete"
    if record["locked"]:
        return "locked"
    if record["interrupted"]:
        return "interrupted"
    if record["prepared"]:
        return "prepared_mx3"
    return "planned"


def tx_bin_index(tx_nm, tx_min_nm, tx_max_nm, tx_bin_width_nm):
    if tx_nm < tx_min_nm or tx_nm > tx_max_nm:
        return None
    idx = int((tx_nm - tx_min_nm) // tx_bin_width_nm)
    n_bins = int((tx_max_nm - tx_min_nm + tx_bin_width_nm - 1e-12) // tx_bin_width_nm)
    return min(max(idx, 0), max(n_bins - 1, 0))


def bin_label(idx, tx_min_nm, tx_bin_width_nm):
    start = tx_min_nm + idx * tx_bin_width_nm
    end = start + tx_bin_width_nm
    return f"{start:.1f}-{end:.1f} nm"


def active_slurm_jobs_for_points(points):
    wanted_job_names = {f"Tz_{format_param(point['Tz'])}" for point in points}
    if not wanted_job_names:
        return []

    try:
        result = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not query Slurm with squeue: {exc}") from exc

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown squeue error"
        raise RuntimeError(f"Could not query Slurm with squeue: {message}")

    active = []
    for line in result.stdout.splitlines():
        job_name = line.strip()
        if job_name in wanted_job_names:
            active.append(job_name)
    return sorted(set(active))


def choose_topup(points, args):
    tx_min_nm = args.topup_tx_min_nm
    tx_max_nm = args.topup_tx_max_nm
    width = args.tx_bin_width_nm
    seed = args.seed

    complete_counts = {}
    candidates_by_bin = {}
    candidate_statuses = {"prepared_mx3", "interrupted", "planned"}

    for point in points:
        tx_nm = nm(point["Tx"])
        idx = tx_bin_index(tx_nm, tx_min_nm, tx_max_nm, width)
        if idx is None:
            continue
        point_status = status(point)
        if point_status == "complete":
            complete_counts[idx] = complete_counts.get(idx, 0) + 1
        elif point_status in candidate_statuses:
            candidates_by_bin.setdefault(idx, []).append(point)

    rng = random.Random(seed)

    def spread_by_tz(candidates, count):
        if count >= len(candidates):
            return sorted(candidates, key=lambda r: (r["Tz"], r["Tx"]))
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda r: (r["Tz"], r["Tx"]))
        if count == 1:
            return [shuffled[len(shuffled) // 2]]
        step = (len(shuffled) - 1) / (count - 1)
        indices = sorted({round(i * step) for i in range(count)})
        cursor = 0
        while len(indices) < count and cursor < len(shuffled):
            if cursor not in indices:
                indices.append(cursor)
            cursor += 1
        return [shuffled[idx] for idx in sorted(indices[:count])]

    selected = []
    for idx in sorted(candidates_by_bin):
        already_complete = complete_counts.get(idx, 0)
        need = max(0, args.target_per_tx_bin - already_complete)
        if need <= 0:
            continue
        selected.extend(spread_by_tz(candidates_by_bin[idx], need))

    selected.sort(key=lambda r: (r["Tx"], r["Tz"]))
    if args.max_submit is not None:
        selected = selected[: args.max_submit]
    return selected, complete_counts, candidates_by_bin


def print_report(points, selected, complete_counts, candidates_by_bin, args):
    counts = {}
    for point in points:
        counts[status(point)] = counts.get(status(point), 0) + 1

    print(f"prefix:          {args.prefix}")
    print(f"points:          {len(points)}")
    for key in sorted(counts):
        print(f"{key + ':':16} {counts[key]}")
    print()
    print(
        "top-up range:    "
        f"Tx={args.topup_tx_min_nm:.1f}..{args.topup_tx_max_nm:.1f} nm"
    )
    print(f"target/bin:      {args.target_per_tx_bin}")
    print(f"max-submit:      {args.max_submit}")
    print(f"selected:        {len(selected)}")
    print()
    print("Tx-bin coverage in top-up range:")
    all_bins = sorted(set(complete_counts) | set(candidates_by_bin))
    for idx in all_bins:
        print(
            f"  {bin_label(idx, args.topup_tx_min_nm, args.tx_bin_width_nm):>13}: "
            f"complete={complete_counts.get(idx, 0):2d}, "
            f"candidates={len(candidates_by_bin.get(idx, [])):3d}"
        )

    if selected:
        print()
        print("selected points [nm]:")
        for point in selected:
            print(f"  Tx={nm(point['Tx']):8.3f}, Tz={nm(point['Tz']):8.3f}")


def submit_points(points, args):
    if not points:
        print("nothing to submit")
        return

    if not args.skip_slurm_check:
        try:
            active_jobs = active_slurm_jobs_for_points(points)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(
                "Refusing to submit while Slurm status is unknown. "
                "Use --skip-slurm-check only after manually confirming there are "
                "no duplicate vx5 jobs queued/running.",
                file=sys.stderr,
            )
            sys.exit(2)
        if active_jobs:
            print(
                "ERROR: matching Slurm jobs are already queued/running: "
                + ", ".join(active_jobs[:10]),
                file=sys.stderr,
            )
            sys.exit(2)

    sys.path.insert(0, str(SIMULATIONS_DIR))
    from swapper import SimulationManager

    manager = SimulationManager(
        main_path=str(Path(args.raw_root)) + "/",
        destination_path=str(Path(args.raw_root)) + "/",
        prefix=args.prefix,
        amumax_bin=args.amumax_bin,
        cuda_module=args.cuda_module,
        use_bad_nodes=args.use_bad_nodes,
    )
    manager.submit_all_simulations(
        params={
            "Tx": [point["Tx"] for point in points],
            "Tz": [point["Tz"] for point in points],
        },
        last_param_name="Tz",
        sbatch=True,
        pairs=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations",
    )
    parser.add_argument("--prefix", default="vx5")
    parser.add_argument("--topup-tx-min-nm", type=float, default=50.0)
    parser.add_argument("--topup-tx-max-nm", type=float, default=100.0)
    parser.add_argument("--tx-bin-width-nm", type=float, default=10.0)
    parser.add_argument("--target-per-tx-bin", type=int, default=6)
    parser.add_argument("--max-submit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--skip-slurm-check", action="store_true")
    parser.add_argument("--amumax-bin", default=None)
    parser.add_argument("--cuda-module", default=None)
    parser.add_argument("--use-bad-nodes", action="store_true")
    args = parser.parse_args()

    if args.max_submit is None and args.submit:
        parser.error("--submit requires --max-submit for this top-up planner.")
    if args.topup_tx_min_nm >= args.topup_tx_max_nm:
        parser.error("--topup-tx-min-nm must be lower than --topup-tx-max-nm.")
    if args.target_per_tx_bin < 1:
        parser.error("--target-per-tx-bin must be positive.")

    points = discover_points(args.raw_root, args.prefix)
    selected, complete_counts, candidates_by_bin = choose_topup(points, args)
    print_report(points, selected, complete_counts, candidates_by_bin, args)

    if not args.submit:
        print()
        print("dry run only; add --submit to submit the selected top-up points")
        return

    submit_points(selected, args)


if __name__ == "__main__":
    main()

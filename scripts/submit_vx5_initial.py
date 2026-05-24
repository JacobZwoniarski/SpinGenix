#!/usr/bin/env python3
"""
Prepare or submit the initial two-parameter Tx/Tz simulation batch.

By default this is a dry run. Pass --submit to actually call sbatch through
SimulationManager.
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


def format_param(value):
    return format(value, ".5g")


def parse_named_param(text, name):
    pattern = re.compile(rf"{name}_?(\d+(?:\.\d+)?e[+-]?\d+)", re.IGNORECASE)
    match = pattern.search(str(text))
    if match:
        return float(match.group(1))
    return None


def parse_point_from_path(path):
    path = Path(path)
    tx = None
    tz = None
    for part in path.parts:
        tx = parse_named_param(part, "Tx") if tx is None else tx
        tz = parse_named_param(part, "Tz") if tz is None else tz
    return tx, tz


def point_key(tx, tz):
    return format_param(tx), format_param(tz)


def point_zarr_path(raw_root, prefix, tx, tz):
    return (
        Path(raw_root)
        / prefix
        / f"Tx_{format_param(tx)}"
        / f"Tz_{format_param(tz)}.zarr"
    )


def point_work_dir(raw_root, prefix, tx):
    return Path(raw_root) / prefix / f"Tx_{format_param(tx)}"


def is_complete_zarr(zarr_path):
    zarr_path = Path(zarr_path)
    return (zarr_path / ".zattrs").exists() and (zarr_path / "m_relaxed" / ".zarray").exists()


def has_status(raw_root, prefix, tx, tz, status):
    work_dir = point_work_dir(raw_root, prefix, tx)
    pattern = f"Tz_{format_param(tz)}.mx3_status.{status}*"
    return any(work_dir.glob(pattern))


def discover_existing_points(raw_root, prefix):
    prefix_dir = Path(raw_root) / prefix
    points = {}
    if not prefix_dir.exists():
        return []

    candidates = []
    candidates.extend(prefix_dir.rglob("*.zarr"))
    candidates.extend(prefix_dir.rglob("*.mx3"))
    candidates.extend(prefix_dir.rglob("*.mx3_status.*"))

    for path in candidates:
        tx, tz = parse_point_from_path(path)
        if tx is None or tz is None:
            continue
        points[point_key(tx, tz)] = (tx, tz)

    return [points[key] for key in sorted(points)]


def generate_lhs_points(count, tx_range, tz_range, seed):
    rng = random.Random(seed)
    tx_slots = list(range(count))
    tz_slots = list(range(count))
    rng.shuffle(tx_slots)
    rng.shuffle(tz_slots)

    tx_span = tx_range[1] - tx_range[0]
    tz_span = tz_range[1] - tz_range[0]
    points = []
    for i in range(count):
        tx_unit = (tx_slots[i] + rng.random()) / count
        tz_unit = (tz_slots[i] + rng.random()) / count
        tx = tx_range[0] + tx_unit * tx_span
        tz = tz_range[0] + tz_unit * tz_span
        points.append((tx, tz))
    return points


def build_target_points(existing_points, target_count, tx_range, tz_range, seed):
    points = {point_key(tx, tz): (tx, tz) for tx, tz in existing_points}
    if len(points) >= target_count:
        return [points[key] for key in sorted(points)]

    needed = target_count - len(points)
    # Oversample so duplicate .5g-formatted keys do not leave us short.
    generated = generate_lhs_points(max(needed * 3, needed + 20), tx_range, tz_range, seed)
    for tx, tz in generated:
        points.setdefault(point_key(tx, tz), (tx, tz))
        if len(points) >= target_count:
            break

    return [points[key] for key in sorted(points)]


def classify_points(points, raw_root, prefix, include_locked=False):
    selected = []
    complete = []
    locked = []
    for tx, tz in points:
        zarr_path = point_zarr_path(raw_root, prefix, tx, tz)
        if is_complete_zarr(zarr_path):
            complete.append((tx, tz))
            continue
        if has_status(raw_root, prefix, tx, tz, "lock") and not include_locked:
            locked.append((tx, tz))
            continue
        selected.append((tx, tz))
    return selected, complete, locked


def nm(value):
    return value * 1e9


def active_slurm_jobs_for_points(points):
    wanted_job_names = {f"Tz_{format_param(tz)}" for _, tz in points}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations",
    )
    parser.add_argument("--prefix", default="vx5")
    parser.add_argument("--target-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tx-min", type=float, default=10.0e-9)
    parser.add_argument("--tx-max", type=float, default=100.0e-9)
    parser.add_argument("--tz-min", type=float, default=10.0e-9)
    parser.add_argument("--tz-max", type=float, default=100.0e-9)
    parser.add_argument(
        "--only-existing",
        action="store_true",
        help="Only submit/restart points already represented by vx5 files/directories.",
    )
    parser.add_argument(
        "--include-locked",
        action="store_true",
        help="Include points with an existing lock status file.",
    )
    parser.add_argument(
        "--max-submit",
        type=int,
        default=None,
        help="Limit how many incomplete points are submitted in this run.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs with sbatch. Omit for dry run.",
    )
    parser.add_argument(
        "--amumax-bin",
        default=None,
        help="Path to an executable amumax binary. Defaults to AMUMAX_BIN or SimulationManager default.",
    )
    parser.add_argument(
        "--cuda-module",
        default=None,
        help="CUDA environment module to load inside submitted Slurm jobs. Defaults to CUDA_MODULE or cuda/12.6.0_560.28.03.",
    )
    parser.add_argument(
        "--use-bad-nodes",
        action="store_true",
        help="Use /mnt/storage_3/home/jakzwo/bad_nodes.txt as an sbatch --exclude list. Disabled by default.",
    )
    parser.add_argument(
        "--skip-slurm-check",
        action="store_true",
        help="Bypass the active-job safety check before --submit.",
    )
    args = parser.parse_args()

    existing = discover_existing_points(args.raw_root, args.prefix)
    if args.only_existing:
        target_points = existing
    else:
        target_points = build_target_points(
            existing,
            args.target_count,
            (args.tx_min, args.tx_max),
            (args.tz_min, args.tz_max),
            args.seed,
        )

    to_submit, complete, locked = classify_points(
        target_points,
        args.raw_root,
        args.prefix,
        include_locked=args.include_locked,
    )
    if args.max_submit is not None:
        to_submit = to_submit[: args.max_submit]

    print(f"prefix:          {args.prefix}")
    print(f"existing points: {len(existing)}")
    print(f"target points:   {len(target_points)}")
    print(f"complete zarr:   {len(complete)}")
    print(f"locked skipped:  {len(locked)}")
    print(f"to submit:       {len(to_submit)}")

    preview = to_submit[:10]
    if preview:
        print("first points to submit [nm]:")
        for tx, tz in preview:
            print(f"  Tx={nm(tx):8.3f}, Tz={nm(tz):8.3f}")

    if not args.submit:
        print("dry run only; add --submit to call sbatch")
        return

    if not to_submit:
        print("nothing to submit")
        return

    if not args.skip_slurm_check:
        try:
            active_jobs = active_slurm_jobs_for_points(to_submit)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(
                "Refusing to submit while Slurm status is unknown. "
                "Retry when squeue works, or pass --skip-slurm-check if you have "
                "manually confirmed no matching vx5 jobs are queued/running.",
                file=sys.stderr,
            )
            sys.exit(2)

        if active_jobs:
            preview = ", ".join(active_jobs[:10])
            if len(active_jobs) > 10:
                preview += f", ... ({len(active_jobs)} total)"
            print(
                "ERROR: matching Slurm jobs are already queued/running: "
                f"{preview}",
                file=sys.stderr,
            )
            print("Refusing to submit duplicate vx5 jobs.", file=sys.stderr)
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
    params = {
        "Tx": [tx for tx, _ in to_submit],
        "Tz": [tz for _, tz in to_submit],
    }
    manager.submit_all_simulations(
        params=params,
        last_param_name="Tz",
        minsim=0,
        maxsim=None,
        sbatch=True,
        pairs=True,
    )


if __name__ == "__main__":
    main()

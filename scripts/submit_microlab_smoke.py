#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microlab_env import load_env_file  # noqa: E402

from run_active_learning import (  # noqa: E402
    DEFAULT_MICROLAB_API_BASE,
    DEFAULT_MICROLAB_PROJECT_ID,
    DEFAULT_SPINGENX_REMOTE_ROOT,
)
from simulations.backends import MicrolabSubmissionBackend, SlurmResources  # noqa: E402
from simulations.swapper import SimulationManager  # noqa: E402


def parse_args(argv=None):
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Generate one SpinGenX .mx3 file and submit it to MicroLab."
    )
    parser.add_argument("--tx-nm", type=float, required=True)
    parser.add_argument("--tz-nm", type=float, required=True)
    parser.add_argument("--simulation-prefix", default="vxAL-smoke")
    parser.add_argument(
        "--simulations-dir",
        default=os.environ.get("SPINGENX_SIMULATIONS_DIR", DEFAULT_SPINGENX_REMOTE_ROOT),
    )
    parser.add_argument("--results-dir", default="results/active_learning")
    parser.add_argument(
        "--microlab-api-base",
        default=os.environ.get("MICROLAB_API_BASE", DEFAULT_MICROLAB_API_BASE),
    )
    parser.add_argument("--microlab-token-env", default="MICROLAB_CLI_TOKEN")
    parser.add_argument(
        "--project-id",
        type=int,
        default=int(os.environ.get("SPINGENX_MICROLAB_PROJECT_ID", DEFAULT_MICROLAB_PROJECT_ID)),
    )
    parser.add_argument("--task-partition", default="proxima")
    parser.add_argument("--task-num-cpus", type=int, default=3)
    parser.add_argument("--task-memory-gb", type=int, default=12)
    parser.add_argument("--task-num-gpus", type=int, default=1)
    parser.add_argument("--task-time-limit", default="24:00:00")
    parser.add_argument("--task-priority", type=int, default=0)
    parser.add_argument("--amumax-version-id", default="default")
    parser.add_argument("--submission-manifest", default=None)
    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument("--wait", action="store_true", help="Wait for the MicroLab task to complete.")
    wait_group.add_argument("--no-wait", action="store_true", help="Submit and return immediately.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check MicroLab CLI auth/API/project configuration without creating a task.",
    )
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--max-wait-hours", type=float, default=24.0)

    args = parser.parse_args(argv)
    if args.submission_manifest is None:
        args.submission_manifest = os.path.join(
            args.results_dir,
            "submissions",
            "submission_manifest.json",
        )
    return args


def build_backend(args, transport=None):
    resources = SlurmResources(
        partition=args.task_partition,
        num_cpus=args.task_num_cpus,
        memory_gb=args.task_memory_gb,
        num_gpus=args.task_num_gpus,
        time_limit=args.task_time_limit,
        priority=args.task_priority,
        amumax_version_id=args.amumax_version_id,
    )
    return MicrolabSubmissionBackend(
        args.microlab_api_base,
        token_env=args.microlab_token_env,
        project_id=args.project_id,
        resources=resources,
        manifest_path=args.submission_manifest,
        transport=transport,
    )


def _record_to_result(record, args) -> Mapping[str, Any]:
    return {
        "backend": record.backend,
        "status": record.status,
        "task_id": record.task_id,
        "slurm_job_id": record.slurm_job_id,
        "output_dir": record.output_dir,
        "mx3_path": record.artifact.mx3_path,
        "zarr_path": record.artifact.zarr_path,
        "project_id": args.project_id,
        "api_base": args.microlab_api_base,
        "submission_manifest": args.submission_manifest,
    }


def _expected_smoke_paths(args) -> tuple[str, str]:
    tx_value = args.tx_nm * 1e-9
    tz_value = args.tz_nm * 1e-9
    tx_dir = f"Tx_{format(tx_value, '.5g')}"
    tz_name = f"Tz_{format(tz_value, '.5g')}"
    mx3_path = Path(args.simulations_dir) / args.simulation_prefix / tx_dir / f"{tz_name}.mx3"
    return str(mx3_path), str(mx3_path.with_suffix(".zarr"))


def _already_complete_result(args) -> Mapping[str, Any]:
    mx3_path, zarr_path = _expected_smoke_paths(args)
    return {
        "backend": "amucontainers",
        "status": "already_complete",
        "task_id": None,
        "slurm_job_id": "",
        "output_dir": zarr_path,
        "mx3_path": mx3_path,
        "zarr_path": zarr_path,
        "project_id": args.project_id,
        "api_base": args.microlab_api_base,
        "submission_manifest": args.submission_manifest,
    }


def main(argv=None, *, transport=None):
    args = parse_args(argv)
    backend = build_backend(args, transport=transport)
    if args.preflight_only:
        result = backend.preflight()
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result

    manager = SimulationManager(
        main_path=args.simulations_dir,
        destination_path=args.simulations_dir,
        prefix=args.simulation_prefix,
        submission_backend=backend,
    )

    records = manager.submit_all_simulations(
        {
            "Tx": [args.tx_nm * 1e-9],
            "Tz": [args.tz_nm * 1e-9],
        },
        last_param_name="Tz",
        pairs=True,
        sbatch=True,
    )
    if not records:
        raise RuntimeError("No MicroLab submission record was produced.")

    record = records[0]
    if record is None:
        result = _already_complete_result(args)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result

    if args.wait:
        completed = backend.wait_for_records(
            records,
            poll_interval=args.poll_interval,
            max_wait_hours=args.max_wait_hours,
        )
        if not completed:
            raise TimeoutError("Timed out waiting for the MicroLab smoke task.")

    result = _record_to_result(record, args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()

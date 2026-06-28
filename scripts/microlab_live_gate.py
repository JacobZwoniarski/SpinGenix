#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_active_learning import (  # noqa: E402
    DEFAULT_MICROLAB_API_BASE,
    DEFAULT_MICROLAB_PROJECT_ID,
    DEFAULT_SPINGENX_REMOTE_ROOT,
)
from scripts import microlab_submission_status, submit_microlab_smoke  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the SpinGenX MicroLab evidence gate: preflight, one smoke "
            "submission, status polling, and zarr validation."
        )
    )
    parser.add_argument("--tx-nm", type=float, default=10.0)
    parser.add_argument("--tz-nm", type=float, default=20.0)
    parser.add_argument("--simulation-prefix", default="vxAL-live-gate")
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
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--max-wait-hours", type=float, default=24.0)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the smoke task and record current status without waiting for completion.",
    )
    parser.add_argument("--evidence-path", default=None)
    args = parser.parse_args(argv)
    if args.submission_manifest is None:
        args.submission_manifest = os.path.join(
            args.results_dir,
            "submissions",
            "submission_manifest.json",
        )
    if args.evidence_path is None:
        args.evidence_path = os.path.join(
            args.results_dir,
            "submissions",
            "microlab_live_gate_evidence.json",
        )
    return args


def _smoke_args(args, *, preflight_only: bool) -> list[str]:
    command = [
        "--tx-nm",
        str(args.tx_nm),
        "--tz-nm",
        str(args.tz_nm),
        "--simulation-prefix",
        args.simulation_prefix,
        "--simulations-dir",
        args.simulations_dir,
        "--results-dir",
        args.results_dir,
        "--microlab-api-base",
        args.microlab_api_base,
        "--microlab-token-env",
        args.microlab_token_env,
        "--project-id",
        str(args.project_id),
        "--task-partition",
        args.task_partition,
        "--task-num-cpus",
        str(args.task_num_cpus),
        "--task-memory-gb",
        str(args.task_memory_gb),
        "--task-num-gpus",
        str(args.task_num_gpus),
        "--task-time-limit",
        args.task_time_limit,
        "--task-priority",
        str(args.task_priority),
        "--amumax-version-id",
        args.amumax_version_id,
        "--submission-manifest",
        args.submission_manifest,
        "--poll-interval",
        str(args.poll_interval),
        "--max-wait-hours",
        str(args.max_wait_hours),
    ]
    if preflight_only:
        command.append("--preflight-only")
    elif args.no_wait:
        command.append("--no-wait")
    else:
        command.append("--wait")
    return command


def _status_args(args) -> list[str]:
    command = [
        "--submission-manifest",
        args.submission_manifest,
        "--microlab-api-base",
        args.microlab_api_base,
        "--microlab-token-env",
        args.microlab_token_env,
        "--project-id",
        str(args.project_id),
        "--check-zarr",
    ]
    return command


def _write_evidence(path_value: str, evidence: Mapping[str, Any]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def main(argv=None, *, transport=None):
    args = parse_args(argv)
    preflight = submit_microlab_smoke.main(
        _smoke_args(args, preflight_only=True),
        transport=transport,
    )
    smoke = submit_microlab_smoke.main(
        _smoke_args(args, preflight_only=False),
        transport=transport,
    )
    status = microlab_submission_status.main(
        _status_args(args),
        transport=transport,
    )
    evidence = {
        "api_base": args.microlab_api_base,
        "project_id": args.project_id,
        "simulations_dir": args.simulations_dir,
        "submission_manifest": args.submission_manifest,
        "preflight": preflight,
        "smoke": smoke,
        "status": status,
    }
    _write_evidence(args.evidence_path, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return evidence


if __name__ == "__main__":
    main()

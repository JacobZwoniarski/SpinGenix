#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microlab_env import load_env_file  # noqa: E402

from run_active_learning import DEFAULT_MICROLAB_API_BASE, DEFAULT_MICROLAB_PROJECT_ID  # noqa: E402
from simulations.backends import MicrolabSubmissionBackend  # noqa: E402


def parse_args(argv=None):
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Query MicroLab statuses for SpinGenX submission manifest entries."
    )
    parser.add_argument(
        "--submission-manifest",
        default=os.path.join(
            "results",
            "active_learning",
            "submissions",
            "submission_manifest.json",
        ),
    )
    parser.add_argument(
        "--microlab-api-base",
        default=os.environ.get("MICROLAB_API_BASE", DEFAULT_MICROLAB_API_BASE),
    )
    parser.add_argument("--microlab-token-env", default="MICROLAB_CLI_TOKEN")
    parser.add_argument(
        "--project-id",
        type=int,
        default=int(os.environ.get("SPINGENX_MICROLAB_PROJECT_ID", DEFAULT_MICROLAB_PROJECT_ID)),
        help="Expected MicroLab project id for manifest tasks.",
    )
    parser.add_argument(
        "--check-zarr",
        action="store_true",
        help="Inspect completed zarr outputs and classify complete/incomplete results.",
    )
    return parser.parse_args(argv)


def _read_manifest(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Submission manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("submissions"), dict):
        raise ValueError(f"Invalid SpinGenX submission manifest: {path}")
    return data


def _write_manifest(path: Path, data: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _is_complete_zarr(path_value: str) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    return (
        (path / ".zattrs").exists()
        and (path / "m_relaxed" / ".zarray").exists()
    )


def _status_with_zarr_check(status: str, zarr_path: str, enabled: bool) -> tuple[str, Optional[bool]]:
    if not enabled or status.upper() not in {"COMPLETED", "DONE", "SUCCESS", "FINISHED"}:
        return status, None
    complete = _is_complete_zarr(zarr_path)
    if complete:
        return "completed_valid_zarr", True
    return "completed_invalid_zarr", False


def main(argv=None, *, transport=None):
    args = parse_args(argv)
    manifest_path = Path(args.submission_manifest)
    manifest = dict(_read_manifest(manifest_path))
    backend = MicrolabSubmissionBackend(
        args.microlab_api_base,
        token_env=args.microlab_token_env,
        transport=transport,
    )

    submissions = []
    summary = Counter()
    submissions_by_hash = manifest["submissions"]
    for param_hash, entry in sorted(submissions_by_hash.items()):
        task_id = entry.get("task_id")
        if task_id in (None, ""):
            status = str(entry.get("status", "missing_task_id"))
            task = {}
        else:
            task = dict(backend.get_task(task_id))
            status = str(task.get("status", "UNKNOWN"))

        returned_project_id = task.get("project_id")
        if returned_project_id is not None and int(returned_project_id) != args.project_id:
            raise RuntimeError(
                f"Task {task_id} belongs to project_id={returned_project_id}; "
                f"expected project_id={args.project_id}"
            )

        output_dir = task.get("output_dir") or entry.get("output_dir") or ""
        zarr_path = entry.get("zarr_path", "") or output_dir
        status, zarr_complete = _status_with_zarr_check(
            status,
            zarr_path,
            args.check_zarr,
        )
        item = {
            "param_hash": param_hash,
            "task_id": task_id,
            "status": status,
            "slurm_job_id": str(task.get("slurm_job_id") or entry.get("slurm_job_id") or ""),
            "output_dir": output_dir,
            "mx3_path": entry.get("mx3_path", ""),
            "zarr_path": zarr_path,
            "prefix": entry.get("prefix", ""),
            "params": entry.get("params", {}),
            "project_id": returned_project_id or entry.get("project_id") or args.project_id,
        }
        if zarr_complete is not None:
            item["zarr_complete"] = zarr_complete
        entry["status"] = status
        entry["slurm_job_id"] = item["slurm_job_id"]
        entry["output_dir"] = output_dir
        entry["zarr_path"] = zarr_path
        entry["project_id"] = item["project_id"]
        if zarr_complete is not None:
            entry["zarr_complete"] = zarr_complete
        entry["last_checked_at"] = int(time.time())
        submissions.append(item)
        summary[status] += 1

    result = {
        "summary": dict(summary),
        "submissions": submissions,
    }
    _write_manifest(manifest_path, manifest)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()

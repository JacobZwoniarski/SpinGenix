#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microlab_env import load_env_file  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-spingenix"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from simulations.backends import SlurmResources, create_submission_backend  # noqa: E402


DEFAULT_MICROLAB_PROJECT_ID = 24
DEFAULT_MICROLAB_PROJECT_URL = "https://amucontainers.orion.zfns.eu.org/dashboard/projects/24"
DEFAULT_MICROLAB_API_BASE = "https://amucontainers.orion.zfns.eu.org/api/v1"
DEFAULT_SPINGENX_REMOTE_ROOT = (
    "/mnt/storage_6/project_data/pl0095-01/mateuszz/microlab/projects/"
    "24_spingenx/workspace/scratch/SpinGenx_remote"
)


def nm_range(min_nm, max_nm):
    return (float(min_nm) * 1e-9, float(max_nm) * 1e-9)


def log_startup(message):
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    print(f"[run_active_learning] {timestamp} {message}", flush=True)


def parse_args(argv=None):
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Run the SpinGenix active-learning loop."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Train/acquire and write acquisition CSV, but do not submit Slurm jobs or mutate the dataset.",
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="Submit selected points to Slurm, wait for completed zarr outputs, and append them to the dataset.",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check MicroLab CLI auth/API/project configuration without training or creating tasks.",
    )

    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--grid-points", type=int, default=40)
    parser.add_argument("--k-new", type=int, default=20)
    parser.add_argument(
        "--max-submit",
        type=int,
        default=None,
        help="Maximum new points to submit per active-learning iteration.",
    )
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--tx-min-nm", type=float, default=10.0)
    parser.add_argument("--tx-max-nm", type=float, default=110.0)
    parser.add_argument("--tz-min-nm", type=float, default=10.0)
    parser.add_argument("--tz-max-nm", type=float, default=110.0)
    parser.add_argument("--acquisition-min-distance-nm", type=float, default=None)

    parser.add_argument("--meta-path", default="data/dataset/meta.h5")
    parser.add_argument("--fields-path", default="data/dataset/fields.npz")
    parser.add_argument("--dataset-dir", default="data/dataset")
    parser.add_argument("--normalizer-path", default="data/dataset/param_normalizer.json")
    parser.add_argument("--registry-dir", default="data/registry")
    parser.add_argument(
        "--dataset-work-dir",
        default=None,
        help=(
            "Mutable per-run dataset/registry copy. Defaults to "
            "<run-dir>/dataset. Ignored with --in-place-dataset."
        ),
    )
    parser.add_argument(
        "--in-place-dataset",
        action="store_true",
        help="Mutate --meta-path/--fields-path/--registry-dir directly. Use only for legacy runs.",
    )
    parser.add_argument("--results-dir", default="results/active_learning")
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Directory for this run's outputs. Defaults to "
            "<results-dir>/run_<UTC timestamp>. Ignored with --in-place-results."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable run directory name under --results-dir, for example al_2nm_v2.",
    )
    parser.add_argument(
        "--in-place-results",
        action="store_true",
        help="Write outputs directly into --results-dir. Use only for legacy layout/debugging.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for model checkpoints. Defaults to <results-dir>/checkpoints.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable model checkpoint writing.",
    )
    parser.add_argument(
        "--checkpoint-every-epoch",
        action="store_true",
        help="Also save a checkpoint after every training epoch, not only after each AL iteration.",
    )
    parser.add_argument(
        "--simulations-dir",
        default=os.environ.get("SPINGENX_SIMULATIONS_DIR", DEFAULT_SPINGENX_REMOTE_ROOT),
    )
    parser.add_argument("--simulation-prefix", default="vxAL")

    parser.add_argument("--poll-interval", type=int, default=300)
    parser.add_argument("--max-wait-hours", type=float, default=24.0)
    parser.add_argument("--amumax-bin", default=None)
    parser.add_argument("--cuda-module", default=None)

    parser.add_argument(
        "--submission-backend",
        default="local-sbatch",
        choices=["local-sbatch", "amucontainers", "dry-run"],
        help="Where to submit prepared .mx3 files.",
    )
    parser.add_argument(
        "--microlab-api-base",
        default=os.environ.get("MICROLAB_API_BASE", DEFAULT_MICROLAB_API_BASE),
        help="MicroLab API base URL, for example https://host/api/v1.",
    )
    parser.add_argument(
        "--microlab-token-env",
        default="MICROLAB_CLI_TOKEN",
        help="Environment variable containing the MicroLab CLI token.",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=int(os.environ.get("SPINGENX_MICROLAB_PROJECT_ID", DEFAULT_MICROLAB_PROJECT_ID)),
        help=(
            "MicroLab project id for submitted tasks. "
            f"Defaults to SpinGenX project {DEFAULT_MICROLAB_PROJECT_ID}: "
            f"{DEFAULT_MICROLAB_PROJECT_URL}"
        ),
    )
    parser.add_argument("--task-partition", default="proxima")
    parser.add_argument("--task-num-cpus", type=int, default=3)
    parser.add_argument("--task-memory-gb", type=int, default=12)
    parser.add_argument("--task-num-gpus", type=int, default=1)
    parser.add_argument("--task-time-limit", default="24:00:00")
    parser.add_argument("--task-priority", type=int, default=0)
    parser.add_argument("--amumax-version-id", default="default")
    parser.add_argument(
        "--submission-manifest",
        default=None,
        help=(
            "JSON manifest used to avoid duplicate MicroLab submissions. "
            "Defaults to <results-dir>/submissions/submission_manifest.json."
        ),
    )

    args = parser.parse_args(argv)
    return args


def build_submission_backend(args, *, transport=None):
    resources = SlurmResources(
        partition=args.task_partition,
        num_cpus=args.task_num_cpus,
        memory_gb=args.task_memory_gb,
        num_gpus=args.task_num_gpus,
        time_limit=args.task_time_limit,
        priority=args.task_priority,
        amumax_version_id=args.amumax_version_id,
    )
    return create_submission_backend(
        args.submission_backend,
        microlab_api_base=args.microlab_api_base,
        microlab_token_env=args.microlab_token_env,
        project_id=args.project_id,
        resources=resources,
        manifest_path=args.submission_manifest,
        transport=transport,
    )


def _utc_run_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    for idx in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{idx:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique dataset work directory under {path.parent}")


def _copy_if_exists(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _copy_registry_if_exists(source, destination_dir):
    source = Path(source)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    if source.is_dir():
        for filename in ("simulations.csv", "simulations.parquet"):
            src = source / filename
            if _copy_if_exists(src, destination_dir / filename):
                copied.append(str(destination_dir / filename))
    elif source.exists():
        destination = destination_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination))

    return copied


def prepare_run_paths(args):
    if args.in_place_results:
        run_dir = Path(args.results_dir)
    else:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        elif args.run_name:
            run_dir = Path(args.results_dir) / args.run_name
        else:
            run_dir = Path(args.results_dir) / f"run_{_utc_run_stamp()}"
        run_dir = _unique_path(run_dir)
        args.results_dir = str(run_dir)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    if args.submission_manifest is None:
        args.submission_manifest = os.path.join(
            args.results_dir,
            "submissions",
            "submission_manifest.json",
        )

    log_startup(f"run outputs: {args.results_dir}")
    return Path(args.results_dir)


def prepare_mutable_run_data(args):
    """
    Protect the seed dataset by copying mutable AL inputs into a per-run workdir.
    The active-learning loop can append new samples without polluting the
    bootstrap dataset used to start future experiments.
    """
    if args.in_place_dataset:
        return None

    work_dir = Path(args.dataset_work_dir) if args.dataset_work_dir else Path(args.results_dir) / "dataset"
    work_dir = _unique_path(work_dir)
    registry_dir = work_dir / "registry"

    copied = {
        "meta": _copy_if_exists(args.meta_path, work_dir / "meta.h5"),
        "fields": _copy_if_exists(args.fields_path, work_dir / "fields.npz"),
        "normalizer": _copy_if_exists(args.normalizer_path, work_dir / "param_normalizer.json"),
        "registry": _copy_registry_if_exists(args.registry_dir, registry_dir),
    }

    args.dataset_dir = str(work_dir)
    args.meta_path = str(work_dir / "meta.h5")
    args.fields_path = str(work_dir / "fields.npz")
    args.normalizer_path = str(work_dir / "param_normalizer.json")
    args.registry_dir = str(registry_dir)

    log_startup(
        "isolated mutable dataset: "
        f"{work_dir} "
        f"(meta={copied['meta']}, fields={copied['fields']}, "
        f"normalizer={copied['normalizer']}, registry_files={len(copied['registry'])})"
    )
    return work_dir


def main(argv=None, *, transport=None):
    args = parse_args(argv)

    if args.preflight_only:
        log_startup("building submission backend for preflight")
        submission_backend = build_submission_backend(args, transport=transport)
        if submission_backend is None or not hasattr(submission_backend, "preflight"):
            raise ValueError("--preflight-only requires --submission-backend amucontainers")
        log_startup("running submission backend preflight")
        result = submission_backend.preflight()
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result

    prepare_run_paths(args)
    log_startup("building submission backend")
    submission_backend = build_submission_backend(args, transport=transport)
    log_startup("submission backend ready")

    log_startup("importing torch and ActiveLearningLoop")
    import torch  # noqa: WPS433
    from active_learning.loop import ActiveLearningLoop  # noqa: WPS433
    log_startup("imports ready")

    device = args.device
    if device == "auto":
        log_startup("checking CUDA availability")
        device = "cuda" if torch.cuda.is_available() else "cpu"

    acquisition_min_distance = None
    if args.acquisition_min_distance_nm is not None:
        acquisition_min_distance = float(args.acquisition_min_distance_nm) * 1e-9

    log_startup("preparing mutable run dataset")
    prepare_mutable_run_data(args)

    log_startup(
        f"mode={'submit' if args.submit else 'dry-run'}, "
        f"device={device}, iterations={args.iterations}, epochs={args.epochs}, "
        f"grid={args.grid_points}x{args.grid_points}, k_new={args.k_new}"
    )

    log_startup("constructing ActiveLearningLoop")
    al = ActiveLearningLoop(
        meta_path=args.meta_path,
        fields_path=args.fields_path,
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        registry_path=args.registry_dir,
        normalizer_path=args.normalizer_path,
        checkpoint_dir=args.checkpoint_dir,
        simulations_dir=args.simulations_dir,
        Tx_range=nm_range(args.tx_min_nm, args.tx_max_nm),
        Tz_range=nm_range(args.tz_min_nm, args.tz_max_nm),
        grid_points=args.grid_points,
        k_new=args.k_new,
        max_submit=args.max_submit,
        mc_samples=args.mc_samples,
        acquisition_min_distance=acquisition_min_distance,
        poll_interval=args.poll_interval,
        max_wait_hours=args.max_wait_hours,
        submit_simulations=args.submit,
        simulation_prefix=args.simulation_prefix,
        amumax_bin=args.amumax_bin,
        cuda_module=args.cuda_module,
        submission_backend=submission_backend,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_every_epoch=args.checkpoint_every_epoch,
    )
    log_startup("starting active learning loop")
    al.run(iterations=args.iterations)
    return None


if __name__ == "__main__":
    main()

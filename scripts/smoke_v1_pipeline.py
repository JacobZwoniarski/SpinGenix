#!/usr/bin/env python3
"""
Smoke checks for the vx5-only SpinGenix V1 pipeline.

This script does not submit Slurm jobs. It validates the local Python
environment, optionally preprocesses one complete vx5 result, validates the
dataset files when present, saves a dataset phase diagram, and runs a tiny
reconstruction baseline smoke test.
"""

import argparse
import getpass
import importlib
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "matplotlib",
    "h5py",
    "tables",
    "zarr",
    "pyarrow",
    "pyzfn",
    "discretisedfield",
]

REQUIRED_METADATA_COLUMNS = {
    "simulation_id",
    "param_hash",
    "split",
    "field_key",
    "Tx_val",
    "Tz_val",
    "MeanMz_signed",
    "MeanMz_abs",
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


def import_smoke():
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            print(f"[env] OK {name}")
        except Exception as exc:
            missing.append(f"{name}: {type(exc).__name__}: {exc}")

    if missing:
        raise RuntimeError("Missing required modules:\n  " + "\n  ".join(missing))


def complete_zarrs(raw_root, prefix):
    root = Path(raw_root) / prefix
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.zarr")
        if (path / ".zattrs").exists() and (path / "m_relaxed" / ".zarray").exists()
    )


def preprocess_smoke(zarr_path):
    from postprocess.preprocess import preprocess_simulation

    field, (tx, tz), metadata = preprocess_simulation(str(zarr_path), target_size=(200, 200))
    print(f"[preprocess] field shape: {field.shape}")
    print(f"[preprocess] Tx={tx:.6g}, Tz={tz:.6g}")
    print(
        "[preprocess] MeanMz_signed={:.6g}, MeanMz_abs={:.6g}, Q={:.6g}".format(
            metadata["MeanMz_signed"],
            metadata["MeanMz_abs"],
            metadata["Q"],
        )
    )
    if field.shape != (200, 200, 3):
        raise RuntimeError(f"Expected field shape (200, 200, 3), got {field.shape}")


def validate_dataset(meta_path, fields_path):
    import numpy as np
    import pandas as pd

    df = pd.read_hdf(meta_path, key="data")
    missing = REQUIRED_METADATA_COLUMNS.difference(df.columns)
    if missing:
        raise RuntimeError(f"Dataset metadata missing columns: {sorted(missing)}")

    with np.load(fields_path, allow_pickle=True) as fields:
        keys = fields.files
        if len(df) != len(keys):
            raise RuntimeError(f"len(meta)={len(df)} != len(fields)={len(keys)}")
        bad = {key: fields[key].shape for key in keys if fields[key].shape != (200, 200, 3)}

    if bad:
        preview = list(bad.items())[:5]
        raise RuntimeError(f"Invalid field shapes: {preview}")

    print(f"[dataset] OK {len(df)} samples")
    return df


def save_dataset_phase_diagram(df, out_dir):
    from active_learning.phase_diagram import plot_dataset_phase_diagram
    import matplotlib.pyplot as plt

    out_path = Path(out_dir) / "phase_diagrams" / "phase_dataset_smoke.png"
    fig, _ = plot_dataset_phase_diagram(df, save_path=out_path)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def training_smoke(
    meta_path,
    fields_path,
    normalizer_path,
    out_dir,
    device,
    epochs,
    batch_size,
    max_samples,
):
    import torch
    from torch.utils.data import Subset

    from active_learning.dataset import MagnetizationDataset
    from active_learning.model import UNetCVAE
    from active_learning.trainer import train_cvae
    from active_learning.visualization import visualize_reconstruction
    import matplotlib.pyplot as plt

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MagnetizationDataset(
        meta_path=meta_path,
        fields_path=fields_path,
        target_size=(200, 200),
        normalizer_path=normalizer_path,
    )
    subset_size = min(max_samples, len(dataset))
    smoke_dataset = Subset(dataset, list(range(subset_size)))

    model = UNetCVAE(spatial_size=200)
    field, params = smoke_dataset[0]
    physical_params = dataset.physical_params(0)
    print(f"[model] normalized params={params.numpy()}, physical params={physical_params}")
    model = model.to(device)
    with torch.no_grad():
        recon, mu, logvar = model(field.unsqueeze(0).to(device), params.unsqueeze(0).to(device))
    print(f"[model] forward output={tuple(recon.shape)}, mu={tuple(mu.shape)}, logvar={tuple(logvar.shape)}")

    if epochs > 0:
        model = train_cvae(
            model,
            smoke_dataset,
            epochs=epochs,
            batch_size=min(batch_size, subset_size),
            device=device,
        )

    checkpoint_dir = Path(out_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "v1_smoke_baseline.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "spatial_size": 200,
            "latent_dim": model.latent_dim,
            "cond_dim": model.cond_dim,
            "param_normalizer": (
                dataset.param_normalizer.to_dict()
                if dataset.param_normalizer is not None
                else None
            ),
            "note": "V1 reconstruction baseline smoke checkpoint; not a validated surrogate.",
        },
        checkpoint_path,
    )
    print(f"[checkpoint] saved {checkpoint_path}")

    with torch.no_grad():
        recon, _, _ = model(field.unsqueeze(0).to(device), params.unsqueeze(0).to(device))
    tx, tz = physical_params
    recon_path = Path(out_dir) / "reconstructions" / "recon_smoke_hsl.png"
    fig, _ = visualize_reconstruction(
        field.cpu().numpy(),
        recon[0].cpu().numpy(),
        tx,
        tz,
        mode="hsl",
        save_path=recon_path,
    )
    plt.close(fig)
    print(f"[reconstruction] saved {recon_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations")
    parser.add_argument("--prefix", default="vx5")
    parser.add_argument("--meta-path", default="data/dataset/meta.h5")
    parser.add_argument("--fields-path", default="data/dataset/fields.npz")
    parser.add_argument("--normalizer-path", default="data/dataset/param_normalizer.json")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--require-dataset", action="store_true")
    args = parser.parse_args()

    import_smoke()

    zarrs = complete_zarrs(args.raw_root, args.prefix)
    print(f"[zarr] complete {args.prefix}: {len(zarrs)}")
    if zarrs:
        preprocess_smoke(zarrs[0])

    meta_path = Path(args.meta_path)
    fields_path = Path(args.fields_path)
    dataset_exists = meta_path.exists() and fields_path.exists()
    if not dataset_exists:
        message = f"Dataset files not found: {meta_path}, {fields_path}"
        if args.require_dataset:
            raise RuntimeError(message)
        print(f"[dataset] SKIP {message}")
        return

    df = validate_dataset(meta_path, fields_path)
    save_dataset_phase_diagram(df, args.out_dir)

    if not args.skip_training:
        training_smoke(
            str(meta_path),
            str(fields_path),
            args.normalizer_path,
            args.out_dir,
            args.device,
            args.train_epochs,
            args.batch_size,
            args.max_samples,
        )


if __name__ == "__main__":
    main()

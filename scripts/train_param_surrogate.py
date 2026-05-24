#!/usr/bin/env python3
"""
Train the first true SpinGenix surrogate: normalized params -> magnetization field.

This is a deterministic V2 baseline, not the older reconstruction-only CVAE path.
It predicts canonical 200x200x3 fields; physical dimensions remain in metadata.
"""

import argparse
import getpass
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from active_learning.dataset import MagnetizationDataset  # noqa: E402
from active_learning.evaluation import (  # noqa: E402
    evaluate_param_surrogate_model,
    save_evaluation_outputs,
    summarize_metrics,
)
from active_learning.param_surrogate import ConditionalResNetDecoder  # noqa: E402
from active_learning.trainer import WeightedMSELoss  # noqa: E402
from active_learning.visualization import visualize_reconstruction  # noqa: E402

HOLDOUT_SPLITS = {"test_holdout", "boundary_holdout", "ood_holdout"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(dataset, val_fraction, seed):
    df = dataset.df.reset_index(drop=True)
    if "split" in df.columns:
        split = df["split"].fillna("train").astype(str)
        train_idx = df.index[split == "train"].tolist()
        val_idx = df.index[split == "val"].tolist()
        holdout_idx = df.index[split.isin(HOLDOUT_SPLITS)].tolist()

        if train_idx:
            if not val_idx and val_fraction > 0 and len(train_idx) > 1:
                rng = random.Random(seed)
                shuffled = train_idx[:]
                rng.shuffle(shuffled)
                val_count = max(1, int(round(len(train_idx) * val_fraction)))
                val_count = min(val_count, len(train_idx) - 1)
                val_idx = sorted(shuffled[:val_count])
                train_idx = sorted(shuffled[val_count:])
            if holdout_idx:
                print(f"held-out samples excluded from training: {len(holdout_idx)}")
            return sorted(train_idx), sorted(val_idx)
        raise ValueError("Dataset has a split column but no train samples.")

    indices = list(range(len(dataset)))
    if len(indices) < 2 or val_fraction <= 0:
        return indices, []

    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_fraction)))
    val_count = min(val_count, len(indices) - 1)
    val_idx = sorted(indices[:val_count])
    train_idx = sorted(indices[val_count:])
    return train_idx, val_idx


def field_unit_norm_penalty(prediction):
    norm = torch.linalg.norm(prediction, dim=1)
    return torch.mean(torch.abs(norm - 1.0))


def run_epoch(model, loader, optimizer, device, rec_loss_fn, norm_weight):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    rows = 0

    for fields, params in tqdm(loader, desc="train" if training else "val", leave=False):
        fields = fields.to(device)
        params = params.to(device)

        with torch.set_grad_enabled(training):
            pred = model(params)
            rec_loss = rec_loss_fn(pred, fields)
            norm_loss = field_unit_norm_penalty(pred)
            loss = rec_loss + norm_weight * norm_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = fields.shape[0]
        total += float(loss.item()) * batch_size
        rows += batch_size

    return total / max(rows, 1)


def save_sample_reconstructions(model, dataset, indices, out_dir, device, count):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) / "reconstructions"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    model.eval()

    for idx in indices[:count]:
        field, params = dataset[idx]
        with torch.no_grad():
            pred = model(params.unsqueeze(0).to(device))[0].cpu().numpy()
        tx, tz = dataset.physical_params(idx)
        path = out_dir / f"param_surrogate_idx{idx}.png"
        fig, _ = visualize_reconstruction(
            field.cpu().numpy(),
            pred,
            tx,
            tz,
            mode="hsl",
            save_path=path,
        )
        plt.close(fig)
        saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", default="data/dataset/meta.h5")
    parser.add_argument("--fields-path", default="data/dataset/fields.npz")
    parser.add_argument("--normalizer-path", default="data/dataset/param_normalizer.json")
    parser.add_argument("--out-dir", default="results/param_surrogate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--norm-weight", type=float, default=0.01)
    parser.add_argument("--save-png", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MagnetizationDataset(
        meta_path=args.meta_path,
        fields_path=args.fields_path,
        target_size=(200, 200),
        normalizer_path=args.normalizer_path,
    )
    if args.max_samples is not None:
        indices = list(range(min(args.max_samples, len(dataset))))
        dataset = Subset(dataset, indices)
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    train_idx, val_idx = split_indices(base_dataset, args.val_fraction, args.seed)
    if args.max_samples is not None:
        allowed = set(range(len(dataset)))
        train_idx = [idx for idx in train_idx if idx in allowed]
        val_idx = [idx for idx in val_idx if idx in allowed]

    train_dataset = Subset(base_dataset, train_idx)
    val_dataset = Subset(base_dataset, val_idx) if val_idx else None

    model = ConditionalResNetDecoder(
        spatial_size=200,
        cond_dim=base_dataset.param_normalizer.dim if base_dataset.param_normalizer else 2,
        hidden_dim=args.hidden_dim,
        base_channels=args.base_channels,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rec_loss_fn = WeightedMSELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, max(len(train_dataset), 1)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset):
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=0,
        )

    history = []
    print(f"device: {args.device}")
    print(f"samples: train={len(train_dataset)}, val={0 if val_dataset is None else len(val_dataset)}")
    print(f"model config: {model.config()}")

    if args.epochs == 0:
        field, params = train_dataset[0]
        with torch.no_grad():
            pred = model(params.unsqueeze(0).to(args.device))
        print(f"forward smoke: target={tuple(field.shape)}, pred={tuple(pred.shape)}")

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            rec_loss_fn,
            args.norm_weight,
        )
        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(
                model,
                val_loader,
                None,
                args.device,
                rec_loss_fn,
                args.norm_weight,
            )
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    checkpoint_path = out_dir / "param_surrogate.pt"
    torch.save(
        {
            "model_class": "ConditionalResNetDecoder",
            "model_config": model.config(),
            "model_state_dict": model.state_dict(),
            "param_normalizer": (
                base_dataset.param_normalizer.to_dict()
                if base_dataset.param_normalizer is not None
                else None
            ),
            "param_columns": list(base_dataset.param_columns),
            "note": "V2 deterministic surrogate baseline: normalized params -> canonical 200x200x3 field.",
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")

    eval_df, predictions = evaluate_param_surrogate_model(
        model,
        base_dataset,
        device=args.device,
        max_samples=args.max_samples,
    )
    summary = summarize_metrics(eval_df)
    eval_paths = save_evaluation_outputs(eval_df, summary, out_dir / "evaluation")
    print(f"eval metrics: {eval_paths['metrics']}")
    print(f"eval summary: {eval_paths['summary']}")
    if "mse_mean" in summary:
        print(f"mse_mean: {summary['mse_mean']:.6g}")
    if "cosine_similarity_mean" in summary:
        print(f"cosine_similarity_mean: {summary['cosine_similarity_mean']:.6g}")

    save_sample_reconstructions(
        model,
        base_dataset,
        list(predictions.keys()),
        out_dir,
        args.device,
        args.save_png,
    )


if __name__ == "__main__":
    main()

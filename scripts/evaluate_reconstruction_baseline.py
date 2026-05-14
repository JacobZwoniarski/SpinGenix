#!/usr/bin/env python3
"""
Evaluate the V1 reconstruction baseline on a prepared dataset split.

This is not surrogate validation: the current CVAE path reconstructs fields
given fields + normalized parameters. Use it to sanity-check reconstruction
quality while V2 true params -> field models are being built.
"""

import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path

import torch

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
    evaluate_reconstruction_model,
    save_evaluation_outputs,
    summarize_metrics,
)
from active_learning.model import UNetCVAE  # noqa: E402
from active_learning.visualization import visualize_reconstruction  # noqa: E402


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNetCVAE(
        spatial_size=checkpoint.get("spatial_size", 200),
        latent_dim=checkpoint.get("latent_dim", 64),
        cond_dim=checkpoint.get("cond_dim", 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def save_reconstruction_pngs(dataset, predictions, out_dir, count):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) / "reconstructions"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for idx in list(predictions.keys())[:count]:
        field, _ = dataset[idx]
        tx, tz = dataset.physical_params(idx)
        fig, _ = visualize_reconstruction(
            field.cpu().numpy(),
            predictions[idx],
            tx,
            tz,
            mode="hsl",
            save_path=out_dir / f"eval_recon_idx{idx}.png",
        )
        plt.close(fig)
        saved.append(out_dir / f"eval_recon_idx{idx}.png")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", default="data/dataset/meta.h5")
    parser.add_argument("--fields-path", default="data/dataset/fields.npz")
    parser.add_argument("--normalizer-path", default="data/dataset/param_normalizer.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="results/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-png", type=int, default=3)
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MagnetizationDataset(
        meta_path=args.meta_path,
        fields_path=args.fields_path,
        target_size=(200, 200),
        normalizer_path=args.normalizer_path,
    )
    model, checkpoint = load_model(args.checkpoint, args.device)

    metrics_df, predictions = evaluate_reconstruction_model(
        model,
        dataset,
        device=args.device,
        max_samples=args.max_samples,
    )
    summary = summarize_metrics(metrics_df)
    paths = save_evaluation_outputs(metrics_df, summary, args.out_dir)
    saved_pngs = save_reconstruction_pngs(dataset, predictions, args.out_dir, args.save_png)

    print("V1 reconstruction baseline evaluation complete.")
    print("Note:", checkpoint.get("note", "not a validated surrogate"))
    print(f"samples: {len(metrics_df)}")
    print(f"mse_mean: {summary.get('mse_mean'):.6g}")
    print(f"mae_mean: {summary.get('mae_mean'):.6g}")
    print(f"cosine_similarity_mean: {summary.get('cosine_similarity_mean'):.6g}")
    print(f"metrics: {paths['metrics']}")
    print(f"summary: {paths['summary']}")
    if saved_pngs:
        print(f"pngs: {len(saved_pngs)} saved under {Path(args.out_dir) / 'reconstructions'}")


if __name__ == "__main__":
    main()

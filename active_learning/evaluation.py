from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import collate_with_meta


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _collate_auto(batch):
    """
    Obsługuje dataset zwracający:
      - (field, params)
      - (field, params, meta)
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")
    if len(batch[0]) == 3:
        return collate_with_meta(batch)
    # 2-tuple fallback
    fields, params = zip(*batch)
    return torch.stack(fields, 0), torch.stack(params, 0)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    model.eval()
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
        collate_fn=_collate_auto,
    )

    mse_c = np.zeros(3, dtype=np.float64)
    mae_c = np.zeros(3, dtype=np.float64)
    n_pix = 0

    meanmz_gt = []
    meanmz_pred = []
    rows = []

    for batch in loader:
        if len(batch) == 2:
            fields, params = batch
            metas = [{} for _ in range(fields.shape[0])]
        else:
            fields, params, metas = batch

        fields = fields.to(device)   # (B,3,H,W)
        params = params.to(device)   # (B,2)

        recon, _, _ = model(fields, params)

        diff = recon - fields
        mse_batch = (diff ** 2).sum(dim=(0, 2, 3))  # (3,)
        mae_batch = diff.abs().sum(dim=(0, 2, 3))   # (3,)

        mse_c += _to_numpy(mse_batch)
        mae_c += _to_numpy(mae_batch)
        n_pix += fields.shape[0] * fields.shape[2] * fields.shape[3]

        gt_mz = fields[:, 2].mean(dim=(1, 2))
        pr_mz = recon[:, 2].mean(dim=(1, 2))
        meanmz_gt.extend(_to_numpy(gt_mz).tolist())
        meanmz_pred.extend(_to_numpy(pr_mz).tolist())

        for i in range(fields.shape[0]):
            meta_i = metas[i] if isinstance(metas, list) else {}
            rows.append({
                **meta_i,
                "mse": float(((recon[i] - fields[i]) ** 2).mean().item()),
                "mae": float((recon[i] - fields[i]).abs().mean().item()),
                "meanmz_gt": float(gt_mz[i].item()),
                "meanmz_pred": float(pr_mz[i].item()),
                "meanmz_abs_err": float((gt_mz[i] - pr_mz[i]).abs().item()),
            })

    mse_c = mse_c / max(1, n_pix)
    mae_c = mae_c / max(1, n_pix)

    out = {
        "mse_total": float(mse_c.mean()),
        "mae_total": float(mae_c.mean()),
        "mse_mx": float(mse_c[0]),
        "mse_my": float(mse_c[1]),
        "mse_mz": float(mse_c[2]),
        "mae_mx": float(mae_c[0]),
        "mae_my": float(mae_c[1]),
        "mae_mz": float(mae_c[2]),
        "meanmz_mae": float(np.mean(np.abs(np.array(meanmz_gt) - np.array(meanmz_pred)))) if len(meanmz_gt) else 0.0,
        "n_samples": int(len(dataset)),
    }

    df = pd.DataFrame(rows)
    return out | {"per_sample": df}


def binned_metrics(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_cols: list[str],
    bins: int = 4,
) -> pd.DataFrame:
    work = df.copy()

    for c in param_cols:
        if c not in work.columns:
            raise ValueError(f"Missing column in per-sample DF: {c}")
        work[f"{c}_bin"] = pd.qcut(work[c], q=bins, duplicates="drop")

    group_cols = [f"{c}_bin" for c in param_cols]
    agg = {m: ["mean", "median", "std", "count", "max"] for m in metric_cols}

    out = work.groupby(group_cols, observed=False).agg(agg).reset_index()
    out.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def save_eval_logs(
    results_dir: str,
    iteration: int,
    summary: dict,
    per_sample_df: pd.DataFrame,
    per_region_df: pd.DataFrame | None = None,
) -> None:
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)

    json_path = os.path.join(results_dir, "logs", f"metrics_iter_{iteration:03d}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    per_sample_path = os.path.join(results_dir, "logs", f"per_sample_iter_{iteration:03d}.csv")
    per_sample_df.to_csv(per_sample_path, index=False)

    if per_region_df is not None:
        per_region_path = os.path.join(results_dir, "logs", f"per_region_iter_{iteration:03d}.csv")
        per_region_df.to_csv(per_region_path, index=False)

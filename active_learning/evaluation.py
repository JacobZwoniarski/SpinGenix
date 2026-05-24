import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from postprocess.preprocess import classify_state, compute_topological_charge


CHANNELS = ("Mx", "My", "Mz")


def _to_hwc(array):
    array = np.asarray(array)
    if array.shape[0] == 3:
        return np.transpose(array, (1, 2, 0))
    return array


def _state_metrics(field_hwc):
    mx = field_hwc[:, :, 0]
    my = field_hwc[:, :, 1]
    mz = field_hwc[:, :, 2]
    mean_mx = float(np.mean(mx))
    mean_my = float(np.mean(my))
    b = float(np.sqrt(mean_mx**2 + mean_my**2))
    q = compute_topological_charge(mx, my, mz)
    return {
        "MeanMx": mean_mx,
        "MeanMy": mean_my,
        "MeanMz_signed": float(np.mean(mz)),
        "MeanMz_abs": float(np.mean(np.abs(mz))),
        "Q": q,
        "State": classify_state(mx, my, mz, b, q),
    }


def reconstruction_metrics(target, prediction):
    target = _to_hwc(target)
    prediction = _to_hwc(prediction)

    diff = prediction - target
    record = {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
    }

    for idx, name in enumerate(CHANNELS):
        channel_diff = diff[:, :, idx]
        record[f"mse_{name}"] = float(np.mean(channel_diff**2))
        record[f"mae_{name}"] = float(np.mean(np.abs(channel_diff)))

    numerator = np.sum(target * prediction, axis=-1)
    denominator = np.linalg.norm(target, axis=-1) * np.linalg.norm(prediction, axis=-1)
    cosine = numerator / np.clip(denominator, 1e-12, None)
    record["cosine_similarity"] = float(np.mean(cosine))

    target_phys = _state_metrics(target)
    pred_phys = _state_metrics(prediction)
    for key in ["MeanMz_signed", "MeanMz_abs", "Q"]:
        record[f"target_{key}"] = target_phys[key]
        record[f"pred_{key}"] = pred_phys[key]
        record[f"abs_error_{key}"] = abs(pred_phys[key] - target_phys[key])

    record["target_State"] = target_phys["State"]
    record["pred_State"] = pred_phys["State"]
    record["state_correct"] = int(target_phys["State"] == pred_phys["State"])
    return record


@torch.no_grad()
def evaluate_reconstruction_model(model, dataset, device="cpu", max_samples=None):
    model = model.to(device)
    model.eval()

    n_samples = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    rows = []
    predictions = {}

    for idx in range(n_samples):
        field, params = dataset[idx]
        recon, _, _ = model(
            field.unsqueeze(0).to(device),
            params.unsqueeze(0).to(device),
        )

        target = field.cpu().numpy()
        prediction = recon[0].cpu().numpy()
        metrics = reconstruction_metrics(target, prediction)

        row = {
            "sample_index": idx,
            **metrics,
        }
        if "simulation_id" in dataset.df.columns:
            row["simulation_id"] = dataset.df.iloc[idx]["simulation_id"]
        if "split" in dataset.df.columns:
            row["split"] = dataset.df.iloc[idx]["split"]
        for column, value in zip(dataset.param_columns, dataset.physical_params(idx)):
            row[column] = float(value)

        rows.append(row)
        predictions[idx] = prediction

    return pd.DataFrame(rows), predictions


@torch.no_grad()
def evaluate_param_surrogate_model(model, dataset, device="cpu", max_samples=None):
    model = model.to(device)
    model.eval()

    n_samples = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    rows = []
    predictions = {}

    for idx in range(n_samples):
        field, params = dataset[idx]
        prediction = model(params.unsqueeze(0).to(device))[0].cpu().numpy()
        target = field.cpu().numpy()
        metrics = reconstruction_metrics(target, prediction)

        row = {
            "sample_index": idx,
            **metrics,
        }
        if "simulation_id" in dataset.df.columns:
            row["simulation_id"] = dataset.df.iloc[idx]["simulation_id"]
        if "split" in dataset.df.columns:
            row["split"] = dataset.df.iloc[idx]["split"]
        for column, value in zip(dataset.param_columns, dataset.physical_params(idx)):
            row[column] = float(value)

        rows.append(row)
        predictions[idx] = prediction

    return pd.DataFrame(rows), predictions


def summarize_metrics(metrics_df):
    numeric = metrics_df.select_dtypes(include=[np.number])
    summary = {}
    for column in numeric.columns:
        summary[f"{column}_mean"] = float(numeric[column].mean())
        summary[f"{column}_std"] = float(numeric[column].std(ddof=0))
    return summary


def summarize_metrics_by_split(metrics_df):
    if "split" not in metrics_df.columns:
        return pd.DataFrame()

    rows = []
    for split, split_df in metrics_df.groupby("split", dropna=False):
        row = {"split": split, "samples": int(len(split_df))}
        row.update(summarize_metrics(split_df))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def worst_reconstruction_rows(metrics_df, top_n=10):
    specs = []
    if "mse" in metrics_df.columns:
        specs.append(("mse", False, "highest_mse"))
    if "cosine_similarity" in metrics_df.columns:
        specs.append(("cosine_similarity", True, "lowest_cosine"))
    if "abs_error_MeanMz_signed" in metrics_df.columns:
        specs.append(("abs_error_MeanMz_signed", False, "highest_signed_mz_error"))
    if "abs_error_Q" in metrics_df.columns:
        specs.append(("abs_error_Q", False, "highest_q_error"))

    rows = []
    for column, ascending, reason in specs:
        ordered = metrics_df.sort_values(column, ascending=ascending).head(top_n)
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            record = row.to_dict()
            record["reason"] = reason
            record["rank"] = rank
            rows.append(record)

    if "state_correct" in metrics_df.columns:
        wrong = metrics_df[metrics_df["state_correct"] == 0].head(top_n)
        for rank, (_, row) in enumerate(wrong.iterrows(), start=1):
            record = row.to_dict()
            record["reason"] = "wrong_state"
            record["rank"] = rank
            rows.append(record)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def save_evaluation_outputs(metrics_df, summary, out_dir, worst_top_n=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "reconstruction_metrics.csv"
    summary_path = out_dir / "reconstruction_summary.json"
    per_split_path = out_dir / "reconstruction_summary_by_split.csv"
    worst_path = out_dir / "worst_reconstructions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    paths = {
        "metrics": metrics_path,
        "summary": summary_path,
    }

    per_split = summarize_metrics_by_split(metrics_df)
    if not per_split.empty:
        per_split.to_csv(per_split_path, index=False)
        paths["summary_by_split"] = per_split_path

    worst = worst_reconstruction_rows(metrics_df, top_n=worst_top_n)
    if not worst.empty:
        worst.to_csv(worst_path, index=False)
        paths["worst"] = worst_path

    return paths

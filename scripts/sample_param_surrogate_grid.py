#!/usr/bin/env python3
"""
Sample a trained params -> field surrogate on a Tx/Tz grid.

Outputs canonical 200x200x3 predicted fields plus metadata containing physical
Tx/Ty/Tz dimensions. The current V2 model does not change raster resolution;
physical size is recovered from metadata.
"""

import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"matplotlib-{getpass.getuser()}"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from active_learning.normalization import ParamNormalizer  # noqa: E402
from active_learning.param_surrogate import ConditionalResNetDecoder  # noqa: E402
from active_learning.phase_diagram import plot_phase_diagram  # noqa: E402
from postprocess.preprocess import compute_topological_charge  # noqa: E402


def nm_to_si(value):
    return float(value) * 1e-9


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_class") != "ConditionalResNetDecoder":
        raise ValueError(
            "Expected a ConditionalResNetDecoder checkpoint from train_param_surrogate.py"
        )
    model = ConditionalResNetDecoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalizer_payload = checkpoint.get("param_normalizer")
    normalizer = ParamNormalizer.from_dict(normalizer_payload) if normalizer_payload else None
    return model, normalizer, checkpoint


def field_metrics(field_hwc):
    mx = field_hwc[:, :, 0]
    my = field_hwc[:, :, 1]
    mz = field_hwc[:, :, 2]
    return {
        "MeanMx": float(np.mean(mx)),
        "MeanMy": float(np.mean(my)),
        "MeanMz_signed": float(np.mean(mz)),
        "MeanMz_abs": float(np.mean(np.abs(mz))),
        "Q": compute_topological_charge(mx, my, mz),
    }


def write_metadata(df, out_dir):
    csv_path = out_dir / "predictions_meta.csv"
    parquet_path = out_dir / "predictions_meta.parquet"
    df.to_csv(csv_path, index=False)
    parquet_error = None
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        parquet_path = None
        parquet_error = f"{type(exc).__name__}: {exc}"
    return csv_path, parquet_path, parquet_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="results/param_surrogate_grid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tx-min-nm", type=float, default=None)
    parser.add_argument("--tx-max-nm", type=float, default=None)
    parser.add_argument("--tz-min-nm", type=float, default=None)
    parser.add_argument("--tz-max-nm", type=float, default=None)
    parser.add_argument("--grid-points", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-fields", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, normalizer, checkpoint = load_checkpoint(args.checkpoint, args.device)
    if normalizer is None:
        raise RuntimeError("Checkpoint has no param_normalizer; cannot sample physical Tx/Tz grid.")

    column_to_idx = {column: idx for idx, column in enumerate(normalizer.param_columns)}
    tx_idx = column_to_idx.get("Tx_val", 0)
    tz_idx = column_to_idx.get("Tz_val", 1)
    tx_min_nm = args.tx_min_nm if args.tx_min_nm is not None else normalizer.mins[tx_idx] * 1e9
    tx_max_nm = args.tx_max_nm if args.tx_max_nm is not None else normalizer.maxs[tx_idx] * 1e9
    tz_min_nm = args.tz_min_nm if args.tz_min_nm is not None else normalizer.mins[tz_idx] * 1e9
    tz_max_nm = args.tz_max_nm if args.tz_max_nm is not None else normalizer.maxs[tz_idx] * 1e9

    tx_values = np.linspace(nm_to_si(tx_min_nm), nm_to_si(tx_max_nm), args.grid_points)
    tz_values = np.linspace(nm_to_si(tz_min_nm), nm_to_si(tz_max_nm), args.grid_points)
    points = [(tx, tz) for tx in tx_values for tz in tz_values]

    rows = []
    fields = {}
    field_index = 0

    for start in range(0, len(points), args.batch_size):
        batch_points = points[start:start + args.batch_size]
        physical = np.asarray(batch_points, dtype=np.float64)
        normalized = normalizer.transform(physical)
        params = torch.tensor(normalized, dtype=torch.float32, device=args.device)
        with torch.no_grad():
            pred = model.sample(params).cpu().numpy()

        for local_idx, (tx, tz) in enumerate(batch_points):
            field_chw = pred[local_idx]
            field_hwc = np.transpose(field_chw, (1, 2, 0)).astype(np.float32)
            field_key = str(field_index)
            if not args.no_fields:
                fields[field_key] = field_hwc

            rows.append({
                "field_key": field_key,
                "Tx_val": tx,
                "Ty_val": tx,
                "Tz_val": tz,
                "Tx_nm": tx * 1e9,
                "Ty_nm": tx * 1e9,
                "Tz_nm": tz * 1e9,
                "target_Nx": model.spatial_size,
                "target_Ny": model.spatial_size,
                "target_Nz": 1,
                "field_representation": "canonical_200x200x3",
                **field_metrics(field_hwc),
            })
            field_index += 1

    meta_df = pd.DataFrame(rows)
    csv_path, parquet_path, parquet_error = write_metadata(meta_df, out_dir)

    fields_path = None
    if not args.no_fields:
        fields_path = out_dir / "predicted_fields.npz"
        np.savez_compressed(fields_path, **fields)

    import matplotlib.pyplot as plt
    phase_specs = [
        ("MeanMz_abs", "Predicted |Mean Mz|", "phase_param_surrogate_abs.png", "viridis"),
        ("MeanMz_signed", "Predicted Mean Mz", "phase_param_surrogate_signed.png", "coolwarm"),
    ]
    for value_col, label, filename, cmap in phase_specs:
        fig, _ = plot_phase_diagram(
            meta_df,
            value_col=value_col,
            title=f"Param Surrogate Phase Diagram ({value_col})",
            colorbar_label=label,
            cmap=cmap,
            save_path=out_dir / filename,
        )
        plt.close(fig)

    print(f"checkpoint note: {checkpoint.get('note', '')}")
    print(f"points: {len(meta_df)}")
    print(f"metadata csv: {csv_path}")
    if parquet_path:
        print(f"metadata parquet: {parquet_path}")
    elif parquet_error:
        print(f"metadata parquet skipped: {parquet_error}")
    if fields_path:
        print(f"fields: {fields_path}")
    print(f"sampled range: Tx={tx_min_nm:.3f}..{tx_max_nm:.3f} nm, Tz={tz_min_nm:.3f}..{tz_max_nm:.3f} nm")
    print(f"phase png: {out_dir / 'phase_param_surrogate_abs.png'}")
    print(f"phase png: {out_dir / 'phase_param_surrogate_signed.png'}")


if __name__ == "__main__":
    main()

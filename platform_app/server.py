#!/usr/bin/env python3
import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib-spingenix-platform"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from active_learning.normalization import ParamNormalizer  # noqa: E402
from active_learning.param_surrogate import ConditionalResNetDecoder  # noqa: E402
from active_learning.visualization import magnetization_to_hsl_rgb  # noqa: E402
from postprocess.preprocess import classify_state, compute_topological_charge  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
SAFE_FILE_ROOTS = ("data", "results")
CHECKPOINT_CACHE = {}
CHECKPOINT_INFO_CACHE = {}


def relpath(path):
    return str(path.relative_to(ROOT))


def safe_repo_path(value):
    requested = (ROOT / unquote(value)).resolve()
    if not requested.is_relative_to(ROOT):
        raise ValueError("Path escapes repository root.")
    relative = requested.relative_to(ROOT)
    if not relative.parts or relative.parts[0] not in SAFE_FILE_ROOTS:
        raise ValueError("Path is not under an allowed data/results root.")
    return requested


def to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def dataset_summary():
    meta_path = ROOT / "data/dataset/meta.h5"
    fields_path = ROOT / "data/dataset/fields.npz"
    normalizer_path = ROOT / "data/dataset/param_normalizer.json"
    if not meta_path.exists():
        return {"available": False, "reason": "data/dataset/meta.h5 missing"}

    df = pd.read_hdf(meta_path, key="data")
    tx_nm = df["Tx_val"].astype(float) * 1e9
    tz_nm = df["Tz_val"].astype(float) * 1e9
    split_counts = (
        df["split"].fillna("unspecified").astype(str).value_counts().sort_index().to_dict()
        if "split" in df.columns
        else {"unspecified": int(len(df))}
    )
    state_counts = (
        df["State"].fillna("unknown").astype(str).value_counts().sort_index().to_dict()
        if "State" in df.columns
        else {}
    )
    bins = [0, 20, 40, 60, 80, 100, 120]
    tx_bins = pd.cut(tx_nm, bins=bins).value_counts().sort_index()
    tz_bins = pd.cut(tz_nm, bins=bins).value_counts().sort_index()
    recent_columns = [
        col for col in ["split", "State", "Tx_val", "Tz_val", "simulation_id"]
        if col in df.columns
    ]
    recent = df.loc[:, recent_columns].tail(12).copy()
    if "Tx_val" in recent:
        recent["Tx_nm"] = recent["Tx_val"].astype(float) * 1e9
        recent = recent.drop(columns=["Tx_val"])
    if "Tz_val" in recent:
        recent["Tz_nm"] = recent["Tz_val"].astype(float) * 1e9
        recent = recent.drop(columns=["Tz_val"])

    return {
        "available": True,
        "samples": int(len(df)),
        "fields_exists": fields_path.exists(),
        "normalizer_exists": normalizer_path.exists(),
        "split_counts": split_counts,
        "state_counts": state_counts,
        "tx_nm": {
            "min": float(tx_nm.min()),
            "max": float(tx_nm.max()),
            "mean": float(tx_nm.mean()),
        },
        "tz_nm": {
            "min": float(tz_nm.min()),
            "max": float(tz_nm.max()),
            "mean": float(tz_nm.mean()),
        },
        "tx_bins": {str(k): int(v) for k, v in tx_bins.items()},
        "tz_bins": {str(k): int(v) for k, v in tz_bins.items()},
        "recent": [
            {key: to_jsonable(value) for key, value in row.items()}
            for row in recent.to_dict(orient="records")
        ],
    }


def read_json_file(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def normalizer_range_nm(normalizer_payload):
    if not normalizer_payload:
        return None

    columns = normalizer_payload.get("param_columns") or []
    mins = normalizer_payload.get("mins") or []
    maxs = normalizer_payload.get("maxs") or []
    ranges = {}
    for name, lower, upper in zip(columns, mins, maxs):
        short = name.replace("_val", "")
        ranges[short] = {
            "min": float(lower) * 1e9,
            "max": float(upper) * 1e9,
        }
    return ranges


def checkpoint_info(checkpoint_path):
    checkpoint_path = checkpoint_path.resolve()
    cache_key = str(checkpoint_path)
    mtime = checkpoint_path.stat().st_mtime
    cached = CHECKPOINT_INFO_CACHE.get(cache_key)
    if cached and cached["mtime"] == mtime:
        return cached["info"]

    info = {
        "path": relpath(checkpoint_path),
        "mtime": mtime,
        "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
    }
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        normalizer_payload = checkpoint.get("param_normalizer")
        info.update({
            "model_class": checkpoint.get("model_class"),
            "model_config": checkpoint.get("model_config"),
            "note": checkpoint.get("note"),
            "param_columns": checkpoint.get("param_columns"),
            "normalizer_range_nm": normalizer_range_nm(normalizer_payload),
        })
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    CHECKPOINT_INFO_CACHE[cache_key] = {"mtime": mtime, "info": info}
    return info


def list_result_runs():
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return []

    runs = {}
    for checkpoint in results_dir.rglob("param_surrogate.pt"):
        run_dir = checkpoint.parent
        runs[relpath(run_dir)] = {
            "path": relpath(run_dir),
            "checkpoint": relpath(checkpoint),
            "checkpoint_info": checkpoint_info(checkpoint),
            "kind": "param_surrogate",
            "summary": read_json_file(run_dir / "evaluation/reconstruction_summary.json"),
            "audit": read_json_file(run_dir / "evaluation/dataset_audit.json"),
            "phase_images": [
                relpath(path) for path in sorted((run_dir / "phase_diagrams").glob("*.png"))
            ],
            "reconstruction_images": [
                relpath(path) for path in sorted((run_dir / "reconstructions").glob("*.png"))[:24]
            ],
        }

    for acquisition in results_dir.rglob("acquisition_iter*.csv"):
        run_dir = acquisition.parents[1]
        run = runs.setdefault(relpath(run_dir), {
            "path": relpath(run_dir),
            "kind": "active_learning",
            "phase_images": [
                relpath(path) for path in sorted((run_dir / "phase_diagrams").glob("*.png"))
            ],
            "reconstruction_images": [],
        })
        run.setdefault("acquisitions", []).append(relpath(acquisition))

    return sorted(runs.values(), key=lambda row: row["path"])


def load_checkpoint(checkpoint_relpath, device="cpu"):
    checkpoint_path = safe_repo_path(checkpoint_relpath)
    cache_key = (str(checkpoint_path), device)
    if cache_key in CHECKPOINT_CACHE:
        return CHECKPOINT_CACHE[cache_key]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_class") != "ConditionalResNetDecoder":
        raise ValueError("Only ConditionalResNetDecoder param_surrogate checkpoints are supported.")

    model = ConditionalResNetDecoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    normalizer_payload = checkpoint.get("param_normalizer")
    normalizer = ParamNormalizer.from_dict(normalizer_payload) if normalizer_payload else None
    CHECKPOINT_CACHE[cache_key] = (model, normalizer, checkpoint)
    return CHECKPOINT_CACHE[cache_key]


def render_hsl_png_base64(field_hwc):
    rgb = magnetization_to_hsl_rgb(field_hwc)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=130)
    ax.imshow(rgb)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def predict_param_surrogate(query):
    checkpoint = query.get("checkpoint", [None])[0]
    if not checkpoint:
        raise ValueError("Missing checkpoint path.")
    tx_nm = float(query.get("tx_nm", [50.0])[0])
    tz_nm = float(query.get("tz_nm", [50.0])[0])
    device = query.get("device", ["cpu"])[0]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, normalizer, checkpoint_payload = load_checkpoint(checkpoint, device=device)
    physical = np.array([tx_nm * 1e-9, tz_nm * 1e-9], dtype=np.float64)
    warnings = []
    normalizer_payload = checkpoint_payload.get("param_normalizer")
    ranges = normalizer_range_nm(normalizer_payload)
    if ranges:
        for label, value in [("Tx", tx_nm), ("Tz", tz_nm)]:
            bounds = ranges.get(label)
            if bounds and not (bounds["min"] <= value <= bounds["max"]):
                warnings.append(
                    f"{label}={value:.3f} nm is outside the checkpoint range "
                    f"{bounds['min']:.3f}-{bounds['max']:.3f} nm."
                )

    normalized = normalizer.transform(physical) if normalizer else physical.astype(np.float32)
    normalized = np.asarray(normalized, dtype=np.float32)[None, :]
    params = torch.from_numpy(normalized).to(device)
    with torch.no_grad():
        field_chw = model.sample(params)[0].detach().cpu().numpy()
    field_hwc = np.transpose(field_chw, (1, 2, 0)).astype(np.float32)
    mx = field_hwc[:, :, 0]
    my = field_hwc[:, :, 1]
    mz = field_hwc[:, :, 2]
    mean_mx = float(np.mean(mx))
    mean_my = float(np.mean(my))
    mean_mz_signed = float(np.mean(mz))
    mean_mz_abs = float(np.mean(np.abs(mz)))
    in_plane_order = float(np.sqrt(mean_mx**2 + mean_my**2))
    topological_charge = float(compute_topological_charge(mx, my, mz))
    state_guess = classify_state(mx, my, mz, in_plane_order, topological_charge)
    return {
        "checkpoint": checkpoint,
        "device": device,
        "tx_nm": tx_nm,
        "tz_nm": tz_nm,
        "warnings": warnings,
        "normalizer_range_nm": ranges,
        "normalized_params": normalized[0].tolist(),
        "state_guess": state_guess,
        "metrics": {
            "MeanMx": mean_mx,
            "MeanMy": mean_my,
            "MeanMz_signed": mean_mz_signed,
            "MeanMz_abs": mean_mz_abs,
            "InPlaneOrder": in_plane_order,
            "Q": topological_charge,
        },
        "model_config": checkpoint_payload.get("model_config"),
        "image_png_base64": render_hsl_png_base64(field_hwc),
    }


class PlatformHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        data = json.dumps(payload, default=to_jsonable).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_static(self, path):
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if not target.is_relative_to(STATIC_DIR) or not target.exists() or target.is_dir():
            self.send_error(404)
            return
        data = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                return self.send_static("index.html")
            if parsed.path.startswith("/static/"):
                return self.send_static(parsed.path.replace("/static/", "", 1))
            if parsed.path == "/api/status":
                return self.send_json({
                    "dataset": dataset_summary(),
                    "runs": list_result_runs(),
                })
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                path = safe_repo_path(query.get("path", [""])[0])
                if not path.exists() or path.is_dir():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/predict":
                return self.send_json(predict_param_surrogate(parse_qs(parsed.query)))
            self.send_error(404)
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PlatformHandler)
    print(f"SpinGenix platform: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

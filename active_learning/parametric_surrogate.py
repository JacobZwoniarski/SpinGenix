"""Runtime utilities for production params -> field surrogate models.

This module is intentionally separate from the older reconstruction CVAE path.
It treats the surrogate as a generator:

    physical parameters -> canonical magnetization field -> scalar metrics

The helpers are written to support the current Tx/Tz model and later
multi-parameter checkpoints through checkpoint metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .param_surrogate import ConditionalResNetDecoder
from .surrogate_schema import (
    SurrogateSchema,
    checkpoint_schema,
    field_metrics,
    field_to_hwc,
)


@dataclass
class LoadedParametricSurrogate:
    """Loaded params -> field model with metadata needed by APIs and CLIs."""

    model: torch.nn.Module
    checkpoint: Mapping[str, Any]
    schema: SurrogateSchema
    device: torch.device


def load_parametric_surrogate(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedParametricSurrogate:
    """Load a ConditionalResNetDecoder checkpoint as a production surrogate."""

    resolved_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model_class = checkpoint.get("model_class")
    if model_class not in (None, "ConditionalResNetDecoder"):
        raise ValueError(
            f"Unsupported surrogate checkpoint model_class={model_class!r}; "
            "expected ConditionalResNetDecoder."
        )

    model_config = dict(checkpoint.get("model_config") or {})
    if not model_config:
        raise ValueError("Checkpoint is missing model_config.")

    model = ConditionalResNetDecoder(**model_config)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint is missing model weights.")
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    return LoadedParametricSurrogate(
        model=model,
        checkpoint=checkpoint,
        schema=checkpoint_schema(checkpoint),
        device=resolved_device,
    )


def _ordered_values(params: Mapping[str, float], columns: Sequence[str]) -> np.ndarray:
    values = []
    missing = []
    for column in columns:
        if column in params:
            values.append(float(params[column]))
        elif column.lower() in params:
            values.append(float(params[column.lower()]))
        elif column.upper() in params:
            values.append(float(params[column.upper()]))
        else:
            missing.append(column)
    if missing:
        raise ValueError(f"Missing required surrogate parameters: {', '.join(missing)}")
    return np.asarray(values, dtype=np.float32)


def _array_from_normalizer_value(value: Any, columns: Sequence[str]) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return np.asarray([float(value[column]) for column in columns], dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((len(columns),), float(array), dtype=np.float32)
    return array.reshape(-1)[: len(columns)]


def normalize_params(
    checkpoint: Mapping[str, Any],
    params: Mapping[str, float],
    columns: Sequence[str],
) -> np.ndarray:
    """Normalize physical parameters according to checkpoint metadata."""

    values = _ordered_values(params, columns)
    normalizer = checkpoint.get("param_normalizer")

    if hasattr(normalizer, "transform"):
        transformed = normalizer.transform(values.reshape(1, -1))
        return np.asarray(transformed, dtype=np.float32).reshape(1, -1)

    if isinstance(normalizer, Mapping):
        mean = _array_from_normalizer_value(
            normalizer.get("mean") or normalizer.get("means"), columns
        )
        std = _array_from_normalizer_value(
            normalizer.get("std") or normalizer.get("stds"), columns
        )
        if mean is not None and std is not None:
            std = np.where(np.abs(std) < 1e-12, 1.0, std)
            return ((values - mean) / std).reshape(1, -1).astype(np.float32)

        min_values = _array_from_normalizer_value(
            normalizer.get("min")
            or normalizer.get("mins")
            or normalizer.get("lower")
            or normalizer.get("lows"),
            columns,
        )
        max_values = _array_from_normalizer_value(
            normalizer.get("max")
            or normalizer.get("maxs")
            or normalizer.get("upper")
            or normalizer.get("highs"),
            columns,
        )
        if min_values is not None and max_values is not None:
            span = np.where(np.abs(max_values - min_values) < 1e-12, 1.0, max_values - min_values)
            return (2.0 * (values - min_values) / span - 1.0).reshape(1, -1).astype(np.float32)

    return values.reshape(1, -1).astype(np.float32)


@torch.no_grad()
def predict_field(
    loaded: LoadedParametricSurrogate,
    params: Mapping[str, float],
) -> np.ndarray:
    """Predict one canonical HWC magnetization field from physical parameters."""

    normalized = normalize_params(
        loaded.checkpoint,
        params,
        loaded.schema.param_columns,
    )
    tensor = torch.as_tensor(normalized, dtype=torch.float32, device=loaded.device)
    if hasattr(loaded.model, "sample"):
        output = loaded.model.sample(tensor)
    else:
        output = loaded.model(tensor)
    return field_to_hwc(output.detach().cpu().numpy())


def predict_metrics(
    loaded: LoadedParametricSurrogate,
    params: Mapping[str, float],
) -> dict[str, float]:
    """Predict a field and return scalar metrics used by phase diagrams."""

    field = predict_field(loaded, params)
    return field_metrics(field)


def range_warnings(
    checkpoint: Mapping[str, Any],
    params: Mapping[str, float],
    columns: Sequence[str],
) -> list[str]:
    """Return warnings for parameters outside recorded training ranges."""

    ranges = (
        checkpoint.get("param_ranges")
        or checkpoint.get("parameter_ranges")
        or checkpoint.get("physical_param_ranges")
        or {}
    )
    warnings: list[str] = []
    for column in columns:
        if column not in params or column not in ranges:
            continue
        bounds = ranges[column]
        if isinstance(bounds, Mapping):
            low = bounds.get("min") or bounds.get("lower")
            high = bounds.get("max") or bounds.get("upper")
        else:
            low, high = bounds[0], bounds[1]
        if low is not None and float(params[column]) < float(low):
            warnings.append(f"{column}={params[column]} is below training range [{low}, {high}].")
        if high is not None and float(params[column]) > float(high):
            warnings.append(f"{column}={params[column]} is above training range [{low}, {high}].")
    return warnings


def default_fixed_params(
    checkpoint: Mapping[str, Any],
    columns: Sequence[str],
) -> dict[str, float]:
    """Choose midpoint/default physical parameters for frozen dimensions."""

    ranges = (
        checkpoint.get("param_ranges")
        or checkpoint.get("parameter_ranges")
        or checkpoint.get("physical_param_ranges")
        or {}
    )
    normalizer = checkpoint.get("param_normalizer") or {}
    means = normalizer.get("mean") or normalizer.get("means") if isinstance(normalizer, Mapping) else None
    defaults: dict[str, float] = {}
    for column in columns:
        if column in ranges:
            bounds = ranges[column]
            if isinstance(bounds, Mapping):
                low = bounds.get("min") or bounds.get("lower")
                high = bounds.get("max") or bounds.get("upper")
            else:
                low, high = bounds[0], bounds[1]
            if low is not None and high is not None:
                defaults[column] = 0.5 * (float(low) + float(high))
                continue
        if isinstance(means, Mapping) and column in means:
            defaults[column] = float(means[column])
        else:
            defaults[column] = 0.0
    return defaults


def sample_phase_records(
    loaded: LoadedParametricSurrogate,
    x_param: str,
    y_param: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    fixed_params: Mapping[str, float] | None = None,
) -> list[dict[str, float]]:
    """Sample model phase metrics over a 2D parameter grid."""

    base = default_fixed_params(loaded.checkpoint, loaded.schema.param_columns)
    if fixed_params:
        base.update({str(key): float(value) for key, value in fixed_params.items()})

    records: list[dict[str, float]] = []
    for y_value in y_values:
        for x_value in x_values:
            params = dict(base)
            params[x_param] = float(x_value)
            params[y_param] = float(y_value)
            metrics = predict_metrics(loaded, params)
            records.append(
                {
                    **{column: float(params[column]) for column in loaded.schema.param_columns},
                    **metrics,
                }
            )
    return records

"""Shared schema helpers for param-to-field surrogate checkpoints and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_FIELD_REPRESENTATION = "canonical_2d_200x200x3"
DEFAULT_TARGET_SHAPE = (200, 200, 3)
DEFAULT_METRIC_COLUMNS = (
    "mean_mx",
    "mean_my",
    "mean_mz",
    "mean_abs_mx",
    "mean_abs_my",
    "mean_abs_mz",
    "mean_norm",
)


@dataclass(frozen=True)
class SurrogateSchema:
    """Runtime description of a trained params -> field surrogate."""

    model_kind: str
    param_columns: tuple[str, ...]
    target_shape: tuple[int, ...]
    field_representation: str = DEFAULT_FIELD_REPRESENTATION
    metric_columns: tuple[str, ...] = DEFAULT_METRIC_COLUMNS

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "param_columns": list(self.param_columns),
            "target_shape": list(self.target_shape),
            "field_representation": self.field_representation,
            "metric_columns": list(self.metric_columns),
        }


def canonical_param_columns(columns: Sequence[str] | None) -> tuple[str, ...]:
    """Return stable param column names for checkpoint/API metadata."""

    if columns:
        return tuple(str(column) for column in columns)
    return ("Tx", "Tz")


def normalize_target_shape(shape: Sequence[int] | None) -> tuple[int, ...]:
    """Normalize checkpoint target shape to HWC form when possible."""

    if not shape:
        return DEFAULT_TARGET_SHAPE
    values = tuple(int(value) for value in shape)
    if len(values) == 3:
        return values
    return values


def build_param_surrogate_schema(
    param_columns: Sequence[str] | None,
    target_shape: Sequence[int] | None = None,
    field_representation: str = DEFAULT_FIELD_REPRESENTATION,
) -> SurrogateSchema:
    return SurrogateSchema(
        model_kind="param_to_field_surrogate",
        param_columns=canonical_param_columns(param_columns),
        target_shape=normalize_target_shape(target_shape),
        field_representation=field_representation,
    )


def checkpoint_schema(checkpoint: Mapping[str, Any]) -> SurrogateSchema:
    """Read a schema from old or new checkpoint metadata."""

    model_config = checkpoint.get("model_config") or {}
    normalizer = checkpoint.get("param_normalizer") or {}
    param_columns = (
        checkpoint.get("param_columns")
        or checkpoint.get("condition_columns")
        or normalizer.get("columns")
        or model_config.get("param_columns")
        or ("Tx", "Tz")
    )
    target_shape = (
        checkpoint.get("target_shape")
        or checkpoint.get("field_shape")
        or model_config.get("target_shape")
        or DEFAULT_TARGET_SHAPE
    )
    return SurrogateSchema(
        model_kind=str(checkpoint.get("model_kind") or "param_to_field_surrogate"),
        param_columns=canonical_param_columns(param_columns),
        target_shape=normalize_target_shape(target_shape),
        field_representation=str(
            checkpoint.get("field_representation") or DEFAULT_FIELD_REPRESENTATION
        ),
        metric_columns=tuple(
            str(column)
            for column in checkpoint.get("metric_columns", DEFAULT_METRIC_COLUMNS)
        ),
    )


def field_to_hwc(field: np.ndarray) -> np.ndarray:
    """Return a field as HWC, accepting CHW or batched BCHW/BHWC arrays."""

    array = np.asarray(field)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D field array, got shape {array.shape}.")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected a 3-channel magnetization field, got {array.shape}.")
    return array


def valid_field_mask(field: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Infer a physical-region mask from the magnetization norm."""

    hwc = field_to_hwc(field)
    return np.linalg.norm(hwc, axis=-1) > float(threshold)


def field_metrics(field: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute scalar phase-map metrics from a predicted or simulated field."""

    hwc = field_to_hwc(field).astype(np.float64, copy=False)
    if mask is None:
        mask = np.ones(hwc.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != hwc.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match field shape {hwc.shape[:2]}."
            )
    values = hwc[mask]
    if values.size == 0:
        values = hwc.reshape(-1, 3)

    norm = np.linalg.norm(values, axis=-1)
    return {
        "mean_mx": float(np.mean(values[:, 0])),
        "mean_my": float(np.mean(values[:, 1])),
        "mean_mz": float(np.mean(values[:, 2])),
        "mean_abs_mx": float(np.mean(np.abs(values[:, 0]))),
        "mean_abs_my": float(np.mean(np.abs(values[:, 1]))),
        "mean_abs_mz": float(np.mean(np.abs(values[:, 2]))),
        "mean_norm": float(np.mean(norm)),
    }


def params_from_query(
    query: Mapping[str, Sequence[str]],
    param_columns: Sequence[str],
) -> dict[str, float]:
    """Parse generic API query params using checkpoint param_columns."""

    parsed: dict[str, float] = {}
    for column in param_columns:
        candidates = (column, column.lower(), column.upper())
        value = None
        for key in candidates:
            if key in query and query[key]:
                value = query[key][0]
                break
        if value is None:
            raise ValueError(f"Missing required parameter: {column}")
        parsed[column] = float(value)
    return parsed

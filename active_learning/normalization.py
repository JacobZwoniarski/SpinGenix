"""
Parameter normalisation for conditional SpinGenix models.

Physical parameters stay in SI units in metadata and the registry. Models see
only normalized values produced by this saved transform.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .registry import PARAM_COLUMNS_V1


class ParamNormalizer:
    def __init__(self, param_columns=PARAM_COLUMNS_V1, mins=None, maxs=None, eps=1e-30):
        self.param_columns = tuple(param_columns)
        self.mins = np.asarray(mins, dtype=np.float64) if mins is not None else None
        self.maxs = np.asarray(maxs, dtype=np.float64) if maxs is not None else None
        self.eps = float(eps)

    @classmethod
    def fit_dataframe(cls, df, param_columns=PARAM_COLUMNS_V1):
        values = df.loc[:, list(param_columns)].to_numpy(dtype=np.float64)
        return cls(
            param_columns=param_columns,
            mins=np.nanmin(values, axis=0),
            maxs=np.nanmax(values, axis=0),
        )

    @classmethod
    def from_dict(cls, payload):
        return cls(
            param_columns=payload["param_columns"],
            mins=payload["mins"],
            maxs=payload["maxs"],
            eps=payload.get("eps", 1e-30),
        )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self):
        self._check_fitted()
        return {
            "type": "minmax_to_minus1_plus1",
            "param_columns": list(self.param_columns),
            "mins": self.mins.tolist(),
            "maxs": self.maxs.tolist(),
            "eps": self.eps,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    def _check_fitted(self):
        if self.mins is None or self.maxs is None:
            raise RuntimeError("ParamNormalizer is not fitted.")

    @property
    def dim(self):
        return len(self.param_columns)

    def transform(self, values):
        self._check_fitted()
        arr = np.asarray(values, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr.reshape(1, -1)

        span = self.maxs - self.mins
        safe_span = np.where(np.abs(span) <= self.eps, 1.0, span)
        transformed = 2.0 * (arr - self.mins) / safe_span - 1.0
        transformed[:, np.abs(span) <= self.eps] = 0.0

        transformed = transformed.astype(np.float32)
        return transformed[0] if squeeze else transformed

    def inverse_transform(self, values):
        self._check_fitted()
        arr = np.asarray(values, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr.reshape(1, -1)

        span = self.maxs - self.mins
        restored = ((arr + 1.0) / 2.0) * span + self.mins
        restored[:, np.abs(span) <= self.eps] = self.mins[np.abs(span) <= self.eps]

        restored = restored.astype(np.float64)
        return restored[0] if squeeze else restored


def default_normalizer_path(meta_path):
    return Path(meta_path).with_name("param_normalizer.json")

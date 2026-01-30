from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _field_to_tensor(field_hw3: np.ndarray) -> torch.Tensor:
    # (H,W,3) -> (3,H,W)
    if field_hw3.ndim != 3 or field_hw3.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3), got {field_hw3.shape}")
    x = np.transpose(field_hw3, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(x)


class MagnetizationDataset(Dataset):
    """
    meta.h5: kolumny min:
      - field_id (int)  (jeśli nie ma -> tworzymy)
      - split (str)     (jeśli nie ma -> 'train')
      - Tx_val, Tz_val  (lub więcej paramów)
    fields.npz: klucze str(field_id) -> (H,W,3) (domyślnie 200x200x3)
    """

    def __init__(
        self,
        meta_path="data/dataset/meta.h5",
        fields_path="data/dataset/fields.npz",
        split: str | None = "train",
        param_cols: list[str] | None = None,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        target_size=(200, 200),
        return_meta: bool = False,
    ):
        self.meta_path = meta_path
        self.fields_path = fields_path
        self.split = split
        self.param_cols = param_cols or ["Tx_val", "Tz_val"]
        self.param_ranges = param_ranges
        self.target_size = tuple(target_size)
        self.return_meta = return_meta

        if os.path.exists(meta_path):
            df = pd.read_hdf(meta_path, key="data")
        else:
            df = pd.DataFrame(columns=["field_id", "split"] + self.param_cols)

        if "field_id" not in df.columns:
            df["field_id"] = np.arange(len(df), dtype=int)
        if "split" not in df.columns:
            df["split"] = "train"

        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.df = df
        self._npz = np.load(fields_path, allow_pickle=False) if os.path.exists(fields_path) else None

    def __len__(self) -> int:
        return len(self.df)

    def _norm(self, name: str, val: float) -> float:
        if self.param_ranges is None:
            return float(val)
        lo, hi = self.param_ranges[name]
        if hi == lo:
            return 0.0
        return float(2.0 * (val - lo) / (hi - lo) - 1.0)  # [-1,1]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx].to_dict()
        fid = int(row["field_id"])

        if self._npz is None:
            raise FileNotFoundError(f"Missing fields file: {self.fields_path}")

        key = str(fid)
        if key not in self._npz.files:
            raise KeyError(f"Field id {fid} not found in {self.fields_path}")

        field = self._npz[key]  # (H,W,3)
        if tuple(field.shape[:2]) != tuple(self.target_size):
            raise ValueError(f"Field has {field.shape[:2]} but expected {self.target_size}. Run preprocess.")

        field_t = _field_to_tensor(field)  # (3,H,W)

        params = [self._norm(c, float(row[c])) for c in self.param_cols]
        params_t = torch.tensor(params, dtype=torch.float32)

        if not self.return_meta:
            return field_t, params_t

        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        return field_t, params_t, meta

    # -------------------------
    # Update / Save
    # -------------------------

    def add_sample(
        self,
        field_hw3: np.ndarray,
        metadata: dict,
        split: str = "train",
    ) -> None:
        """
        Dodaje próbkę do datasetu na dysku (meta + fields).
        """
        os.makedirs(os.path.dirname(self.meta_path) or ".", exist_ok=True)

        # normalize field shape
        if isinstance(field_hw3, np.ndarray) and field_hw3.ndim == 4 and field_hw3.shape[0] == 1:
            field_hw3 = field_hw3[0]
        if field_hw3.ndim != 3 or field_hw3.shape[-1] != 3:
            raise ValueError(f"add_sample expects (H,W,3), got {field_hw3.shape}")
        if tuple(field_hw3.shape[:2]) != tuple(self.target_size):
            raise ValueError(f"add_sample got {field_hw3.shape[:2]} but expected {self.target_size}")

        # load full meta (unfiltered)
        if os.path.exists(self.meta_path):
            full = pd.read_hdf(self.meta_path, key="data")
        else:
            full = pd.DataFrame(columns=["field_id", "split"] + self.param_cols)

        if "field_id" not in full.columns:
            full["field_id"] = np.arange(len(full), dtype=int)
        if "split" not in full.columns:
            full["split"] = "train"

        next_id = int(full["field_id"].max() + 1) if len(full) > 0 else 0

        row = {"field_id": next_id, "split": split}
        for c in self.param_cols:
            if c not in metadata:
                raise KeyError(f"metadata missing required param column: {c}")
            row[c] = float(metadata[c])

        for k, v in metadata.items():
            if k not in row:
                row[k] = v

        full = pd.concat([full, pd.DataFrame([row])], ignore_index=True)

        # update fields.npz (UWAGA: npz nie wspiera dopisywania — przepisujemy całość)
        fields_dict = {}
        if os.path.exists(self.fields_path):
            old = np.load(self.fields_path, allow_pickle=False)
            for k in old.files:
                fields_dict[k] = old[k]

        fields_dict[str(next_id)] = field_hw3.astype(np.float32)
        np.savez_compressed(self.fields_path, **fields_dict)
        full.to_hdf(self.meta_path, key="data", mode="w")

        self._npz = np.load(self.fields_path, allow_pickle=False)

    def save(self, *args, **kwargs) -> None:
        """
        No-op (dla kompatybilności z loop.py).
        add_sample zapisuje od razu na dysk.
        """
        return


def collate_with_meta(batch):
    """
    Obsługuje oba przypadki:
      - list[(field, params)]
      - list[(field, params, meta)]
    """
    if len(batch[0]) == 2:
        fields, params = zip(*batch)
        return torch.stack(fields, dim=0), torch.stack(params, dim=0)

    fields, params, metas = zip(*batch)
    return torch.stack(fields, dim=0), torch.stack(params, dim=0), list(metas)

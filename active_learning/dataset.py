import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py

from .normalization import ParamNormalizer, default_normalizer_path
from .registry import PARAM_COLUMNS_V1


class MagnetizationDataset(Dataset):
    """
    Unified dataset for magnetization fields + Tx/Tz parameters.
    Loads preprocessed NPZ + HDF5 metadata.
    
    After Active Learning iterations, new samples can be appended
    and re-saved in the same format.
    """

    def __init__(
        self,
        meta_path,
        fields_path,
        device="cpu",
        target_size=(300, 300),
        normalize=True,
        param_columns=PARAM_COLUMNS_V1,
        param_normalizer=None,
        normalizer_path="auto",
        fit_param_normalizer=True,
    ):
        """
        meta_path   : path to HDF5 file with metadata
        fields_path : path to NPZ with magnetization arrays
        target_size : expected size (H, W)
        """
        self.device = device
        self.target_size = target_size
        self.normalize = normalize
        self.param_columns = tuple(param_columns)

        # ------------------------------------------------------
        # Load metadata
        # ------------------------------------------------------
        self.df = pd.read_hdf(meta_path, key="data")

        # Clean metadata
        self.df = self.df.reset_index(drop=True)

        # Required columns check
        missing_params = [column for column in self.param_columns if column not in self.df.columns]
        if missing_params:
            raise KeyError(f"Dataset metadata missing parameter columns: {missing_params}")

        self.normalizer_path = None
        if normalizer_path == "auto":
            self.normalizer_path = default_normalizer_path(meta_path)
        elif normalizer_path:
            self.normalizer_path = normalizer_path

        if param_normalizer is not None:
            self.param_normalizer = param_normalizer
        elif self.normalizer_path and os.path.exists(self.normalizer_path):
            self.param_normalizer = ParamNormalizer.load(self.normalizer_path)
        elif fit_param_normalizer:
            self.param_normalizer = ParamNormalizer.fit_dataframe(self.df, self.param_columns)
        else:
            self.param_normalizer = None
        if (
            self.param_normalizer is not None
            and tuple(self.param_normalizer.param_columns) != self.param_columns
        ):
            raise ValueError(
                "Param normalizer columns "
                f"{self.param_normalizer.param_columns} do not match dataset columns "
                f"{self.param_columns}"
            )

        # ------------------------------------------------------
        # Load fields from NPZ
        # ------------------------------------------------------
        with np.load(fields_path, allow_pickle=True) as loaded_fields:
            self.field_arrays = {
                key: loaded_fields[key]
                for key in loaded_fields.files
            }

        # Ensure matching lengths
        assert len(self.df) == len(self.field_arrays), \
            f"Metadata length {len(self.df)} != fields length {len(self.field_arrays)}"

        # ------------------------------------------------------
        # Create a mapping index → array key
        # ------------------------------------------------------
        self.keys = sorted(self.field_arrays.keys(), key=lambda x: int(x))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Returns:
            field_tensor (3, H, W)
            params_tensor (Tx, Tz)
        """

        # ------------------------------
        # Load magnetization field
        # ------------------------------
        arr = self.field_arrays[self.keys[idx]]

        # Expected shapes:
        # current version: (1, H, W, 3)
        # or (H, W, 3)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]   # remove leading dimension

        assert arr.shape[-1] == 3, f"Expected 3 channels, got {arr.shape}"

        H, W, C = arr.shape

        # Resize if needed
        if (H, W) != self.target_size:
            raise ValueError(
                f"Field size {H,W} does not match expected {self.target_size}. "
                "Adjust your preprocessing or UNet padding."
            )

        # Convert to PyTorch (C, H, W)
        field = torch.tensor(arr, dtype=torch.float32)
        field = field.permute(2, 0, 1)

        if self.normalize:
            # Magnetization should already be in [-1, 1], but ensure numeric safety
            field = torch.clamp(field, -1.0, 1.0)

        params = torch.tensor(self.normalized_params(idx), dtype=torch.float32)

        return field, params

    def physical_params(self, idx):
        row = self.df.iloc[idx]
        return np.array([float(row[column]) for column in self.param_columns], dtype=np.float64)

    def normalized_params(self, idx):
        params = self.physical_params(idx)
        if self.param_normalizer is None:
            return params.astype(np.float32)
        return self.param_normalizer.transform(params)

    def refit_param_normalizer(self):
        self.param_normalizer = ParamNormalizer.fit_dataframe(self.df, self.param_columns)
        return self.param_normalizer

    # ------------------------------------------------------------------
    # ------------------------ SAVING / UPDATING -----------------------
    # ------------------------------------------------------------------

    def add_sample(self, field_array, Tx, Tz, extra_metadata=None):
        """
        Add one new (field, Tx, Tz) sample to the dataset.
        Does NOT save immediately to disk.
        """
        idx = len(self.df)

        # Append metadata row
        meta_row = {
            "Tx_val": Tx,
            "Tz_val": Tz
        }
        if extra_metadata:
            meta_row.update(extra_metadata)

        self.df = pd.concat([self.df, pd.DataFrame([meta_row])], ignore_index=True)

        # Store field
        key = str(idx)
        self.field_arrays[key] = np.asarray(field_array, dtype=np.float32)
        self.keys.append(key)

    def save(self, meta_path, fields_path, normalizer_path="auto"):
        """
        Save updated dataset to disk in same format.
        """
        # Save metadata
        self.df.to_hdf(meta_path, key="data", mode="w", format="table")

        # Save fields
        np.savez_compressed(
            fields_path,
            **{str(i): self.field_arrays[str(i)] for i in range(len(self.df))}
        )

        if normalizer_path:
            normalizer_path = (
                default_normalizer_path(meta_path)
                if normalizer_path == "auto"
                else normalizer_path
            )
            self.refit_param_normalizer().save(normalizer_path)

        print(f"✔ Saved dataset:\n   meta → {meta_path}\n   fields → {fields_path}")

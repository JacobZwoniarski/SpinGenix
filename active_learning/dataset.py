import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py


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
        normalize=True
    ):
        """
        meta_path   : path to HDF5 file with metadata
        fields_path : path to NPZ with magnetization arrays
        target_size : expected size (H, W)
        """
        self.device = device
        self.target_size = target_size
        self.normalize = normalize

        # ------------------------------------------------------
        # Load metadata
        # ------------------------------------------------------
        self.df = pd.read_hdf(meta_path, key="data")

        # Clean metadata
        self.df = self.df.reset_index(drop=True)

        # Required columns check
        assert "Tx_val" in self.df.columns
        assert "Tz_val" in self.df.columns

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

        # ------------------------------
        # Load parameters (Tx, Tz)
        # ------------------------------
        row = self.df.iloc[idx]
        Tx = float(row["Tx_val"])
        Tz = float(row["Tz_val"])

        params = torch.tensor([Tx, Tz], dtype=torch.float32)

        return field, params

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

    def save(self, meta_path, fields_path):
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

        print(f"✔ Saved dataset:\n   meta → {meta_path}\n   fields → {fields_path}")

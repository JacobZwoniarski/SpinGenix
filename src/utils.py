# src/utils.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class RealMagnetizationDatasetNew(Dataset):
    def __init__(self, df, scaling_factors):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.scaling_factors = scaling_factors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        Aex_scaled = row["Aex"]   * self.scaling_factors["Aex"]
        Msat_scaled = row["Msat"] * self.scaling_factors["Msat"]
        Tx_scaled   = row["Tx_val"] * self.scaling_factors["Tx"]
        Tz_scaled   = row["Tz_val"] * self.scaling_factors["Tz"]
        p = np.array([Aex_scaled, Msat_scaled, Tx_scaled, Tz_scaled], dtype=np.float32)
        mag_field = row["field"]
        if mag_field.ndim == 4:
            mag_field = mag_field[0]
        if mag_field.shape[-1] == 3:
            mag_field = np.transpose(mag_field, (2, 0, 1))
        return torch.FloatTensor(p), torch.FloatTensor(mag_field)

def load_dataframe_and_fields(meta_in, fields_in):
    df = pd.read_hdf(meta_in, key="data")
    for col in ["Aex", "Msat", "Tx_val", "Tz_val"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fields_npz = np.load(fields_in)
    fields_list = []
    for i in range(len(df)):
        arr = fields_npz[str(i)]
        fields_list.append(arr)
    df["field"] = fields_list
    return df

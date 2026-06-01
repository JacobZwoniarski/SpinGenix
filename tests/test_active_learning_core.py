import pandas as pd
import numpy as np

from active_learning.acquisition import select_top_k
from active_learning.loop import ActiveLearningLoop, ACQUISITION_COLUMNS
from active_learning.registry import PARAM_COLUMNS_V1, compute_param_hash


class DummyDataset:
    def __init__(self, splits):
        self.df = pd.DataFrame({"split": splits})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # pragma: no cover - Subset is inspected by indices only
        raise AssertionError(f"unexpected sample access: {idx}")


def test_training_subset_uses_train_only():
    dataset = DummyDataset([
        "train",
        "val",
        "test_holdout",
        "boundary_holdout",
        "train",
    ])
    al = ActiveLearningLoop(
        registry_path=None,
        submit_simulations=False,
        device="cpu",
    )

    subset = al.training_subset(dataset)

    assert list(subset.indices) == [0, 4]


def test_acquisition_excludes_holdout_hash_and_existing_point():
    tx_grid = np.array([10e-9, 20e-9, 30e-9])
    tz_grid = np.array([10e-9, 20e-9, 30e-9])
    uncertainty = np.array([
        [10.0, 9.0, 8.0],
        [7.0, 6.0, 5.0],
        [4.0, 3.0, 2.0],
    ])
    holdout_hash = compute_param_hash(
        {"Tx_val": 10e-9, "Tz_val": 10e-9},
        param_names=PARAM_COLUMNS_V1,
    )
    existing_points = np.array([[10e-9, 20e-9]])

    new_tx, new_tz = select_top_k(
        uncertainty,
        tx_grid,
        tz_grid,
        K=2,
        min_distance=0.0,
        existing_points=existing_points,
        excluded_hashes={holdout_hash},
        param_columns=PARAM_COLUMNS_V1,
    )

    assert (10e-9, 10e-9) not in set(zip(new_tx, new_tz))
    assert (10e-9, 20e-9) not in set(zip(new_tx, new_tz))
    assert list(zip(new_tx, new_tz)) == [(10e-9, 30e-9), (20e-9, 10e-9)]


def test_save_acquisition_writes_headers_for_empty_selection(tmp_path):
    al = ActiveLearningLoop(
        registry_path=None,
        submit_simulations=False,
        results_dir=str(tmp_path),
        device="cpu",
    )

    path = al.save_acquisition(
        iteration=1,
        tx_values=[],
        tz_values=[],
        expected_paths=[],
        dry_run=True,
    )
    saved = pd.read_csv(path)

    assert list(saved.columns) == ACQUISITION_COLUMNS
    assert saved.empty

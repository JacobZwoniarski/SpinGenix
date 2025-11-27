import numpy as np
import pandas as pd
import os

from SpinGenix.active_learning.loop import ActiveLearningLoop
from SpinGenix.active_learning.dataset import MagnetizationDataset
from SpinGenix.active_learning.model import UNetCVAE
from SpinGenix.active_learning.trainer import train_cvae
from SpinGenix.active_learning.uncertainty import compute_uncertainty_map
from SpinGenix.active_learning.acquisition import select_top_k


# ============================================================
# Dummy Active Learning (safe dry-run)
# ============================================================

class DummyAL(ActiveLearningLoop):

    def wait_for_simulations(self, _):
        # Pretend that all simulations finished instantly
        print("[DummyAL] Skipping simulation waiting step.")
        return True

    def run(self, iterations=1):
        """
        Run AL loop without submitting or reading real simulations.
        This validates training, uncertainty, and acquisition flow.
        """
        for it in range(iterations):
            print("\n" + "=" * 60)
            print(f"ACTIVE LEARNING ITERATION (DUMMY) {it+1}/{iterations}")
            print("=" * 60)

            # ------------------------------------------------------
            # 1. Load dataset
            # ------------------------------------------------------
            dataset = MagnetizationDataset(
                meta_path=self.meta_path,
                fields_path=self.fields_path,
                device=self.device,
                target_size=(200, 200)
            )
            print(f"[DummyAL] Loaded dataset with {len(dataset)} samples.")

            # ------------------------------------------------------
            # 2. Train CVAE
            # ------------------------------------------------------
            print("[DummyAL] Training CVAE...")
            model = UNetCVAE(spatial_size=200)
            model = train_cvae(
                model,
                dataset,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device
            )

            # ------------------------------------------------------
            # 3. Compute uncertainty map
            # ------------------------------------------------------
            Tx_grid, Tz_grid = self.generate_grid()
            print(f"[DummyAL] Computing uncertainty on {self.grid_points}×{self.grid_points} grid...")

            U = compute_uncertainty_map(
                model,
                Tx_grid,
                Tz_grid,
                mc_samples=10,
                device=self.device
            )

            # ------------------------------------------------------
            # 4. Acquisition: select top-K uncertain points
            # ------------------------------------------------------
            new_Tx, new_Tz = select_top_k(U, Tx_grid, Tz_grid, K=self.k_new)
            print(f"[DummyAL] Selected {len(new_Tx)} new points (dummy mode).")

            print("[DummyAL] Iteration complete (no simulations submitted).")

        print("\n[DUMMY AL] All iterations completed successfully.")


# ============================================================
# MAIN: Prepare fake dataset + run dummy AL
# ============================================================

if __name__ == "__main__":

    # Ensure dataset directory exists
    os.makedirs("data/dataset", exist_ok=True)

    # Create minimal fake metadata
    df = pd.DataFrame([{
        "Tx_val": 50e-9,
        "Tz_val": 30e-9,
        "Q": 0.0,
        "b": 0.0,
        "MeanMx": 0.1,
        "MeanMy": 0.1,
        "MeanMz": 0.8,
        "State": "out-of-plane",
        "Aex": 1.3e-11,
        "Msat": 8e5
    }])
    df.to_hdf("data/dataset/meta.h5", key="data", mode="w")

    # Create fake field tensor (200×200×3)
    np.savez_compressed("data/dataset/fields.npz", **{"0": np.zeros((200,200,3))})

    # Run dummy AL iteration
    al = DummyAL()
    al.run(iterations=1)

    print("\nSUCCESS: Dummy AL iteration dry-run executed.")
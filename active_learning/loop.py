import os
import time
import torch
import numpy as np
from datetime import datetime

from .dataset import MagnetizationDataset
from .trainer import train_cvae
from .model import UNetCVAE
from .uncertainty import compute_uncertainty_map
from .acquisition import select_top_k

# visualization + phase diagram
from .visualization import visualize_reconstruction
from .phase_diagram import predict_phase_mz, plot_phase_diagram, plot_dataset_phase_diagram

# External modules
from simulations.swapper import SimulationManager
from postprocess.preprocess import preprocess_simulation

import matplotlib.pyplot as plt


class ActiveLearningLoop:

    def __init__(
        self,
        # PATHS
        meta_path="data/dataset/meta.h5",
        fields_path="data/dataset/fields.npz",
        dataset_dir="data/dataset/",
        raw_dir="data/raw/",
        processed_dir="data/processed/",
        results_dir="results/",
        simulations_dir="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/",
        # AL PARAMS
        Tx_range=(10e-9, 100e-9),
        Tz_range=(10e-9, 100e-9),
        grid_points=40,
        k_new=20,
        acquisition_min_distance=None,
        poll_interval=300,
        max_wait_hours=24,
        # TRAINING PARAMS
        epochs=20,
        batch_size=8,
        lr=1e-4,
        device="cuda"
    ):
        self.meta_path = meta_path
        self.fields_path = fields_path
        self.dataset_dir = dataset_dir
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.results_dir = results_dir
        self.simulations_dir = simulations_dir

        # AL settings
        self.Tx_range = Tx_range
        self.Tz_range = Tz_range
        self.grid_points = grid_points
        self.k_new = k_new
        self.acquisition_min_distance = acquisition_min_distance
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours

        # Training
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        # HPC simulation manager
        self.sim_manager = SimulationManager(
            main_path=simulations_dir,
            destination_path=simulations_dir,
            prefix="vxAL"
        )

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def generate_grid(self):
        Tx_grid = np.linspace(self.Tx_range[0], self.Tx_range[1], self.grid_points)
        Tz_grid = np.linspace(self.Tz_range[0], self.Tz_range[1], self.grid_points)
        return Tx_grid, Tz_grid

    @staticmethod
    def format_param(value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return format(float(value), ".5g")
        return str(value)

    def wait_for_simulations(self, expected_paths):
        start = time.time()
        max_wait = self.max_wait_hours * 3600

        print(f"[AL] Waiting for {len(expected_paths)} simulation results...")

        while True:
            ready = [p for p in expected_paths if os.path.exists(p)]

            if len(ready) == len(expected_paths):
                print("[AL] All simulations finished.")
                return True

            elapsed = time.time() - start
            if elapsed > max_wait:
                print("[AL] ERROR: Max waiting time exceeded.")
                return False

            print(f"[AL] {len(ready)}/{len(expected_paths)} ready... "
                  f"sleeping {self.poll_interval} seconds.")
            time.sleep(self.poll_interval)

    # ---------------------------------------------------------------
    # Save visualizations
    # ---------------------------------------------------------------

    def save_reconstructions(self, model, dataset, iteration, N=5):
        out_dir = os.path.join(self.results_dir, "reconstructions")
        os.makedirs(out_dir, exist_ok=True)

        model.eval()
        indices = np.random.choice(len(dataset), size=min(N, len(dataset)), replace=False)

        for idx in indices:
            field, params = dataset[idx]
            with torch.no_grad():
                recon, mu, logvar = model(
                    field.unsqueeze(0).to(self.device),
                    params.unsqueeze(0).to(self.device)
                )

            orig = field.cpu().numpy()
            pred = recon[0].cpu().numpy()
            Tx, Tz = params.cpu().numpy()

            # HSL mode
            fig, _ = visualize_reconstruction(orig, pred, Tx, Tz, mode="hsl")
            fig.savefig(
                os.path.join(out_dir, f"recon_iter{iteration}_idx{idx}_hsl.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Components mode
            fig, _ = visualize_reconstruction(orig, pred, Tx, Tz, mode="components")
            fig.savefig(
                os.path.join(out_dir, f"recon_iter{iteration}_idx{idx}_components.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

    def save_dataset_phase_diagram(self, dataset, iteration):
        out_dir = os.path.join(self.results_dir, "phase_diagrams")
        os.makedirs(out_dir, exist_ok=True)

        fig, _ = plot_dataset_phase_diagram(
            dataset.df,
            save_path=os.path.join(out_dir, f"phase_dataset_iter{iteration}.png"),
        )
        plt.close(fig)

    def save_phase_diagram(self, model, iteration):
        out_dir = os.path.join(self.results_dir, "phase_diagrams")
        os.makedirs(out_dir, exist_ok=True)

        Tx_grid, Tz_grid = self.generate_grid()
        df = predict_phase_mz(model, Tx_grid, Tz_grid, device=self.device)

        fig, _ = plot_phase_diagram(
            df,
            title="Model-Predicted Phase Diagram",
            save_path=os.path.join(out_dir, f"phase_model_iter{iteration}.png"),
        )
        plt.close(fig)

    # ---------------------------------------------------------------
    # Main AL loop
    # ---------------------------------------------------------------

    def run(self, iterations=5):

        for it in range(iterations):
            print("\n" + "="*60)
            print(f"ACTIVE LEARNING ITERATION {it+1}/{iterations}")
            print("="*60)

            # 1. Load dataset
            dataset = MagnetizationDataset(
                meta_path=self.meta_path,
                fields_path=self.fields_path,
                device=self.device,
                target_size=(200,200)
            )
            print(f"[AL] Loaded dataset: {len(dataset)} samples.")
            self.save_dataset_phase_diagram(dataset, iteration=it+1)

            # 2. Train model
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
            # NEW: save reconstructions + phase diagram
            # ------------------------------------------------------
            print("[AL] Saving reconstructions...")
            self.save_reconstructions(model, dataset, iteration=it+1)

            print("[AL] Saving model-based phase diagram...")
            self.save_phase_diagram(model, iteration=it+1)

            # ------------------------------------------------------
            # 3. Compute uncertainty map
            # ------------------------------------------------------
            Tx_grid, Tz_grid = self.generate_grid()
            print(f"[AL] Computing uncertainty on {self.grid_points}×{self.grid_points} grid...")
            U = compute_uncertainty_map(
                model,
                Tx_grid,
                Tz_grid,
                mc_samples=10,
                device=self.device
            )

            # 4. Select K new points
            if self.acquisition_min_distance is None:
                tx_step = abs(Tx_grid[1] - Tx_grid[0]) if len(Tx_grid) > 1 else 0.0
                tz_step = abs(Tz_grid[1] - Tz_grid[0]) if len(Tz_grid) > 1 else 0.0
                min_distance = 0.5 * min(tx_step, tz_step)
            else:
                min_distance = self.acquisition_min_distance

            existing_points = dataset.df[["Tx_val", "Tz_val"]].to_numpy(dtype=float)
            new_Tx, new_Tz = select_top_k(
                U,
                Tx_grid,
                Tz_grid,
                K=self.k_new,
                min_distance=min_distance,
                existing_points=existing_points,
            )
            print(f"[AL] Selected {len(new_Tx)} new points for iteration {it+1}.")

            # 5. Submit simulations
            params = {"Tx": new_Tx, "Tz": new_Tz}

            print("[AL] Submitting new simulations to HPC...")
            self.sim_manager.submit_all_simulations(
                params,
                last_param_name="Tz",
                pairs=True,
                sbatch=True
            )

            expected_paths = [
                os.path.join(
                    self.simulations_dir,
                    self.sim_manager.prefix,
                    f"Tx_{self.format_param(Tx)}",
                    f"Tz_{self.format_param(Tz)}.zarr",
                )
                for Tx, Tz in zip(new_Tx, new_Tz)
            ]

            # 6. Wait
            self.wait_for_simulations(expected_paths)

            # 7. Preprocess results
            print("[AL] Preprocessing new results...")
            for zarr_path in expected_paths:
                field, (Tx, Tz), metadata = preprocess_simulation(zarr_path)
                dataset.add_sample(field, Tx, Tz, metadata)

            dataset.save(self.meta_path, self.fields_path)
            print("[AL] Dataset updated.")

        print("\n[AL] Active Learning completed successfully.")

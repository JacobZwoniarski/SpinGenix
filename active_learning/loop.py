import os
import json
import shutil
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from torch.utils.data import Subset

from .dataset import MagnetizationDataset
from .trainer import train_param_surrogate
from .param_surrogate import ConditionalResNetDecoder
from .surrogate_schema import DEFAULT_FIELD_REPRESENTATION, build_param_surrogate_schema
from .acquisition import select_top_k
from .registry import (
    HOLDOUT_SPLITS,
    PARAM_COLUMNS_V1,
    make_registry_row,
    registry_points_and_hashes,
    upsert_registry,
)

# visualization + phase diagram
from .visualization import visualize_reconstruction, visualize_reconstruction_components
from .phase_diagram import (
    predict_phase_mz,
    plot_phase_diagram,
    plot_dataset_phase_comparison,
    plot_dataset_phase_diagram,
)

# External modules
from simulations.swapper import SimulationManager

import matplotlib.pyplot as plt


ACQUISITION_COLUMNS = [
    "iteration",
    "rank",
    "Tx_val",
    "Tz_val",
    "Tx_nm",
    "Tz_nm",
    "expected_zarr_path",
    "dry_run",
]


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
        registry_path="data/registry",
        normalizer_path=None,
        checkpoint_dir=None,
        simulations_dir="/mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/",
        # AL PARAMS
        Tx_range=(10e-9, 100e-9),
        Tz_range=(10e-9, 100e-9),
        grid_points=40,
        k_new=20,
        max_submit=None,
        mc_samples=10,
        acquisition_min_distance=None,
        poll_interval=300,
        max_wait_hours=24,
        submit_simulations=True,
        simulation_prefix="vxAL",
        amumax_bin=None,
        cuda_module=None,
        submission_backend=None,
        # TRAINING PARAMS
        epochs=20,
        batch_size=8,
        lr=1e-4,
        device="cuda",
        param_columns=PARAM_COLUMNS_V1,
        save_checkpoints=True,
        checkpoint_every_epoch=False,
    ):
        self.meta_path = meta_path
        self.fields_path = fields_path
        self.dataset_dir = dataset_dir
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.results_dir = results_dir
        self.registry_path = registry_path
        self.normalizer_path = normalizer_path or os.path.join(dataset_dir, "param_normalizer.json")
        self.checkpoint_dir = checkpoint_dir or os.path.join(results_dir, "checkpoints")
        self.simulations_dir = simulations_dir
        self.param_columns = tuple(param_columns)
        self.save_checkpoints = bool(save_checkpoints)
        self.checkpoint_every_epoch = bool(checkpoint_every_epoch)

        # AL settings
        self.Tx_range = Tx_range
        self.Tz_range = Tz_range
        self.grid_points = grid_points
        self.k_new = k_new
        self.max_submit = max_submit
        self.mc_samples = mc_samples
        self.acquisition_min_distance = acquisition_min_distance
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours
        self.submit_simulations = submit_simulations
        self.simulation_prefix = simulation_prefix

        # Training
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        # HPC simulation manager
        self.sim_manager = SimulationManager(
            main_path=simulations_dir,
            destination_path=simulations_dir,
            prefix=simulation_prefix,
            amumax_bin=amumax_bin,
            cuda_module=cuda_module,
            submission_backend=submission_backend,
        )

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def generate_grid(self):
        Tx_grid = np.linspace(self.Tx_range[0], self.Tx_range[1], self.grid_points)
        Tz_grid = np.linspace(self.Tz_range[0], self.Tz_range[1], self.grid_points)
        return Tx_grid, Tz_grid

    @staticmethod
    def log(message):
        print(message, flush=True)

    @staticmethod
    def format_param(value):
        if isinstance(value, (int, float, np.integer, np.floating)):
            return format(float(value), ".5g")
        return str(value)

    def limit_selected_points(self, tx_values, tz_values):
        if self.max_submit is None:
            return tx_values, tz_values
        limit = max(0, int(self.max_submit))
        return tx_values[:limit], tz_values[:limit]

    def wait_for_simulations(self, expected_paths):
        start = time.time()
        max_wait = self.max_wait_hours * 3600

        self.log(f"[AL] Waiting for {len(expected_paths)} simulation results...")

        while True:
            ready = [
                p for p in expected_paths
                if self.sim_manager.check_simulation_completion(p)[0]
            ]

            if len(ready) == len(expected_paths):
                self.log("[AL] All simulations finished.")
                return True

            elapsed = time.time() - start
            if elapsed > max_wait:
                self.log("[AL] ERROR: Max waiting time exceeded.")
                return False

            self.log(
                f"[AL] {len(ready)}/{len(expected_paths)} complete... "
                f"sleeping {self.poll_interval} seconds."
            )
            time.sleep(self.poll_interval)

    def load_registry_exclusions(self):
        if not self.registry_path:
            return None, set()

        points, hashes = registry_points_and_hashes(
            self.registry_path,
            param_columns=self.param_columns,
        )
        if points is not None:
            self.log(f"[AL] Registry exclusions: {len(points)} points, {len(hashes)} hashes.")
        return points, hashes

    def upsert_registry_rows(self, rows):
        if not self.registry_path or not rows:
            return
        _, paths = upsert_registry(rows, registry_dir=self.registry_path)
        self.log(f"[AL] Registry updated: {paths['csv']}")

    def checkpoint_metadata(self, *, model, dataset, iteration, epoch, loss, stage):
        split_counts = {}
        if hasattr(dataset, "df") and "split" in dataset.df.columns:
            split_counts = {
                str(key): int(value)
                for key, value in dataset.df["split"].fillna("<missing>").value_counts().items()
            }

        normalizer = getattr(dataset, "param_normalizer", None)
        normalizer_payload = None
        if normalizer is not None:
            try:
                normalizer_payload = normalizer.to_dict()
            except RuntimeError:
                normalizer_payload = None

        return {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "stage": stage,
            "iteration": int(iteration),
            "epoch": None if epoch is None else int(epoch),
            "loss": None if loss is None else float(loss),
            "model": {
                "class": model.__class__.__name__,
                "spatial_size": getattr(model, "spatial_size", None),
                "latent_dim": getattr(model, "latent_dim", None),
                "cond_dim": getattr(model, "cond_dim", None),
            },
            "training": {
                "epochs": int(self.epochs),
                "batch_size": int(self.batch_size),
                "lr": float(self.lr),
                "device": str(self.device),
            },
            "active_learning": {
                "Tx_range": [float(self.Tx_range[0]), float(self.Tx_range[1])],
                "Tz_range": [float(self.Tz_range[0]), float(self.Tz_range[1])],
                "grid_points": int(self.grid_points),
                "k_new": int(self.k_new),
                "max_submit": None if self.max_submit is None else int(self.max_submit),
                "mc_samples": int(self.mc_samples),
                "acquisition_min_distance": self.acquisition_min_distance,
                "param_columns": list(self.param_columns),
            },
            "paths": {
                "meta_path": self.meta_path,
                "fields_path": self.fields_path,
                "dataset_dir": self.dataset_dir,
                "registry_path": self.registry_path,
                "normalizer_path": self.normalizer_path,
                "results_dir": self.results_dir,
                "simulations_dir": self.simulations_dir,
                "simulation_prefix": self.simulation_prefix,
            },
            "dataset": {
                "samples": int(len(dataset)) if hasattr(dataset, "__len__") else None,
                "split_counts": split_counts,
                "param_normalizer": normalizer_payload,
            },
        }

    @staticmethod
    def checkpoint_rng_state():
        state = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    def save_model_checkpoint(self, *, model, optimizer, dataset, iteration, epoch, loss, stage):
        if not self.save_checkpoints:
            return None

        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        epoch_part = "final" if epoch is None else f"epoch_{int(epoch):03d}"
        stem = f"iter_{int(iteration):03d}_{epoch_part}_{stage}"
        checkpoint_path = checkpoint_dir / f"{stem}.pt"
        metadata_path = checkpoint_dir / f"{stem}.json"
        latest_path = checkpoint_dir / "latest.pt"
        latest_metadata_path = checkpoint_dir / "latest.json"

        metadata = self.checkpoint_metadata(
            model=model,
            dataset=dataset,
            iteration=iteration,
            epoch=epoch,
            loss=loss,
            stage=stage,
        )
        normalizer_payload = metadata.get("dataset", {}).get("param_normalizer")
        schema = build_param_surrogate_schema(
            self.param_columns,
            target_shape=(200, 200, 3),
            field_representation=DEFAULT_FIELD_REPRESENTATION,
        )
        param_ranges = {}
        if normalizer_payload:
            for column, lower, upper in zip(
                normalizer_payload.get("param_columns", []),
                normalizer_payload.get("mins", []),
                normalizer_payload.get("maxs", []),
            ):
                param_ranges[str(column)] = {"min": float(lower), "max": float(upper)}

        payload = {
            "schema_version": 2,
            **schema.to_checkpoint_dict(),
            "model_class": model.__class__.__name__,
            "model_config": model.config() if hasattr(model, "config") else {},
            "model_state_dict": model.state_dict(),
            "param_normalizer": normalizer_payload,
            "param_columns": list(self.param_columns),
            "param_ranges": param_ranges,
            "metadata": metadata,
            "training": metadata.get("training", {}),
            "active_learning": metadata.get("active_learning", {}),
            "optimizer_state_dict": (
                optimizer.state_dict()
                if hasattr(optimizer, "state_dict")
                else optimizer
            ),
            "rng_state": self.checkpoint_rng_state(),
            "note": "Active-learning params -> canonical 200x200x3 field surrogate checkpoint.",
        }

        tmp_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        tmp_metadata = metadata_path.with_suffix(".json.tmp")
        torch.save(payload, tmp_checkpoint)
        with open(tmp_metadata, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        os.replace(tmp_checkpoint, checkpoint_path)
        os.replace(tmp_metadata, metadata_path)
        shutil.copy2(checkpoint_path, latest_path)
        shutil.copy2(metadata_path, latest_metadata_path)
        if stage == "final":
            run_dir = Path(self.results_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, run_dir / "param_surrogate.pt")
            shutil.copy2(metadata_path, run_dir / "param_surrogate.json")
        self.log(f"[AL] Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def training_subset(self, dataset):
        if "split" not in dataset.df.columns:
            self.log("[AL] Dataset has no split column; training on all samples.")
            return dataset

        split = dataset.df["split"].fillna("train").astype(str)
        train_idx = dataset.df.index[split == "train"].tolist()
        excluded = dataset.df.index[split.isin(HOLDOUT_SPLITS)].tolist()
        val_idx = dataset.df.index[split == "val"].tolist()

        if not train_idx:
            raise ValueError("Dataset split column exists, but no train samples are available.")

        self.log(
            "[AL] Training split: "
            f"train={len(train_idx)}, val_excluded={len(val_idx)}, "
            f"strict_holdout_excluded={len(excluded)}"
        )
        return Subset(dataset, train_idx)

    def registry_rows_for_points(
        self,
        tx_values,
        tz_values,
        zarr_paths,
        status,
        iteration,
        field_keys=None,
        submission_records=None,
    ):
        rows = []
        field_keys = field_keys or [None] * len(zarr_paths)
        submission_records = submission_records or [None] * len(zarr_paths)
        for Tx, Tz, zarr_path, field_key, submission_record in zip(
            tx_values,
            tz_values,
            zarr_paths,
            field_keys,
            submission_records,
        ):
            extra = None
            if submission_record is not None:
                extra = {
                    "submission_backend": submission_record.backend,
                    "microlab_task_id": "" if submission_record.task_id is None else submission_record.task_id,
                    "slurm_job_id": submission_record.slurm_job_id or "",
                    "task_status": submission_record.status,
                    "output_dir": submission_record.output_dir or "",
                }
            rows.append(make_registry_row(
                params={"Tx_val": Tx, "Tz_val": Tz},
                prefix=self.sim_manager.prefix,
                split="train",
                source="active_learning",
                status=status,
                iteration=iteration,
                param_names=self.param_columns,
                zarr_path=zarr_path,
                field_path=self.fields_path if status == "done" else None,
                field_key=field_key,
                extra=extra,
            ))
        return rows

    def wait_for_submission_records(self, submission_records):
        backend = getattr(self.sim_manager, "submission_backend", None)
        if not submission_records or backend is None or not hasattr(backend, "wait_for_records"):
            return True

        self.log(f"[AL] Waiting for {len(submission_records)} MicroLab task(s)...")
        return backend.wait_for_records(
            submission_records,
            poll_interval=self.poll_interval,
            max_wait_hours=self.max_wait_hours,
        )

    def save_acquisition(self, iteration, tx_values, tz_values, expected_paths, dry_run=False):
        out_dir = os.path.join(self.results_dir, "acquisition")
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for rank, (Tx, Tz, path) in enumerate(zip(tx_values, tz_values, expected_paths), start=1):
            rows.append({
                "iteration": iteration,
                "rank": rank,
                "Tx_val": float(Tx),
                "Tz_val": float(Tz),
                "Tx_nm": float(Tx) * 1e9,
                "Tz_nm": float(Tz) * 1e9,
                "expected_zarr_path": path,
                "dry_run": bool(dry_run),
            })
        path = os.path.join(out_dir, f"acquisition_iter{iteration}.csv")
        pd.DataFrame(rows, columns=ACQUISITION_COLUMNS).to_csv(path, index=False)
        self.log(f"[AL] Acquisition saved: {path}")
        return path

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
                pred_tensor = model.sample(params.unsqueeze(0).to(self.device))

            orig = field.cpu().numpy()
            pred = pred_tensor[0].cpu().numpy()
            Tx, Tz = dataset.physical_params(int(idx))

            # HSL mode
            fig, _ = visualize_reconstruction(orig, pred, Tx, Tz, mode="hsl")
            fig.savefig(
                os.path.join(out_dir, f"recon_iter{iteration}_idx{idx}_hsl.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Components mode
            fig, _ = visualize_reconstruction_components(orig, pred, Tx, Tz)
            fig.savefig(
                os.path.join(out_dir, f"recon_iter{iteration}_idx{idx}_components.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

    def save_dataset_phase_diagram(self, dataset, iteration):
        out_dir = os.path.join(self.results_dir, "phase_diagrams")
        os.makedirs(out_dir, exist_ok=True)

        tx_range_nm = (self.Tx_range[0] * 1e9, self.Tx_range[1] * 1e9)
        tz_range_nm = (self.Tz_range[0] * 1e9, self.Tz_range[1] * 1e9)
        fig, _ = plot_dataset_phase_diagram(
            dataset.df,
            tx_range_nm=tx_range_nm,
            tz_range_nm=tz_range_nm,
            save_path=os.path.join(out_dir, f"phase_dataset_iter{iteration}.png"),
        )
        plt.close(fig)

        fig, _ = plot_dataset_phase_comparison(
            dataset.df,
            tx_range_nm=tx_range_nm,
            tz_range_nm=tz_range_nm,
            save_path=os.path.join(out_dir, f"phase_dataset_comparison_iter{iteration}.png"),
        )
        plt.close(fig)

    def save_phase_diagram(self, model, iteration):
        out_dir = os.path.join(self.results_dir, "phase_diagrams")
        os.makedirs(out_dir, exist_ok=True)

        Tx_grid, Tz_grid = self.generate_grid()
        dataset = MagnetizationDataset(
            meta_path=self.meta_path,
            fields_path=self.fields_path,
            device=self.device,
            target_size=(200, 200),
            param_columns=self.param_columns,
            normalizer_path=self.normalizer_path,
        )
        df = predict_phase_mz(
            model,
            Tx_grid,
            Tz_grid,
            device=self.device,
            param_normalizer=dataset.param_normalizer,
        )

        fig, _ = plot_phase_diagram(
            df,
            title="Param-Surrogate Phase Diagram",
            save_path=os.path.join(out_dir, f"phase_model_iter{iteration}.png"),
        )
        plt.close(fig)

    def compute_coverage_acquisition(self, dataset, Tx_grid, Tz_grid):
        """Score candidate points by distance from existing simulated points."""

        existing_points = dataset.df[["Tx_val", "Tz_val"]].to_numpy(dtype=float)
        registry_points, _ = self.load_registry_exclusions()
        if registry_points is not None and len(registry_points):
            existing_points = np.vstack([existing_points, registry_points])

        U = np.zeros((len(Tx_grid), len(Tz_grid)), dtype=np.float32)
        if existing_points.size == 0:
            U.fill(1.0)
            return U

        for i, Tx in enumerate(Tx_grid):
            for j, Tz in enumerate(Tz_grid):
                diff = existing_points[:, :2] - np.asarray([Tx, Tz], dtype=float)
                distances = np.linalg.norm(diff, axis=1)
                U[i, j] = float(np.min(distances))

        max_value = float(np.max(U))
        if max_value > 0:
            U /= max_value
        return U

    # ---------------------------------------------------------------
    # Main AL loop
    # ---------------------------------------------------------------

    def run(self, iterations=5):

        for it in range(iterations):
            self.log("\n" + "="*60)
            self.log(f"ACTIVE LEARNING ITERATION {it+1}/{iterations}")
            self.log("="*60)

            # 1. Load dataset
            dataset = MagnetizationDataset(
                meta_path=self.meta_path,
                fields_path=self.fields_path,
                device=self.device,
                target_size=(200,200),
                param_columns=self.param_columns,
                normalizer_path=self.normalizer_path,
            )
            self.log(f"[AL] Loaded dataset: {len(dataset)} samples.")
            self.save_dataset_phase_diagram(dataset, iteration=it+1)

            # 2. Train production params -> field surrogate
            cond_dim = dataset.param_normalizer.dim if dataset.param_normalizer is not None else len(self.param_columns)
            model = ConditionalResNetDecoder(spatial_size=200, cond_dim=cond_dim)
            train_dataset = self.training_subset(dataset)

            def checkpoint_callback(model, optimizer, epoch, loss):
                if self.checkpoint_every_epoch:
                    self.save_model_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        dataset=dataset,
                        iteration=it + 1,
                        epoch=epoch,
                        loss=loss,
                        stage="epoch",
                    )

            model = train_param_surrogate(
                model,
                train_dataset,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
                checkpoint_callback=checkpoint_callback,
            )
            self.save_model_checkpoint(
                model=model,
                optimizer=getattr(model, "_last_optimizer_state_dict", None),
                dataset=dataset,
                iteration=it + 1,
                epoch=self.epochs,
                loss=getattr(model, "_last_epoch_loss", None),
                stage="final",
            )

            # ------------------------------------------------------
            # NEW: save reconstructions + phase diagram
            # ------------------------------------------------------
            self.log("[AL] Saving reconstructions...")
            self.save_reconstructions(model, dataset, iteration=it+1)

            self.log("[AL] Saving model-based phase diagram...")
            self.save_phase_diagram(model, iteration=it+1)

            # ------------------------------------------------------
            # 3. Compute acquisition score map
            # ------------------------------------------------------
            Tx_grid, Tz_grid = self.generate_grid()
            self.log(f"[AL] Computing coverage acquisition on {self.grid_points}x{self.grid_points} grid...")
            U = self.compute_coverage_acquisition(dataset, Tx_grid, Tz_grid)

            # 4. Select K new points
            if self.acquisition_min_distance is None:
                tx_step = abs(Tx_grid[1] - Tx_grid[0]) if len(Tx_grid) > 1 else 0.0
                tz_step = abs(Tz_grid[1] - Tz_grid[0]) if len(Tz_grid) > 1 else 0.0
                min_distance = 0.5 * min(tx_step, tz_step)
            else:
                min_distance = self.acquisition_min_distance

            existing_points = dataset.df[["Tx_val", "Tz_val"]].to_numpy(dtype=float)
            registry_points, registry_hashes = self.load_registry_exclusions()
            if registry_points is not None and len(registry_points):
                existing_points = np.vstack([existing_points, registry_points])
            new_Tx, new_Tz = select_top_k(
                U,
                Tx_grid,
                Tz_grid,
                K=self.k_new,
                min_distance=min_distance,
                existing_points=existing_points,
                excluded_hashes=registry_hashes,
                param_columns=self.param_columns,
            )
            new_Tx, new_Tz = self.limit_selected_points(new_Tx, new_Tz)
            self.log(f"[AL] Selected {len(new_Tx)} new points for iteration {it+1}.")

            # 5. Submit simulations
            params = {"Tx": new_Tx, "Tz": new_Tz}

            expected_paths = [
                os.path.join(
                    self.simulations_dir,
                    self.sim_manager.prefix,
                    f"Tx_{self.format_param(Tx)}",
                    f"Tz_{self.format_param(Tz)}.zarr",
                )
                for Tx, Tz in zip(new_Tx, new_Tz)
            ]
            self.save_acquisition(
                iteration=it + 1,
                tx_values=new_Tx,
                tz_values=new_Tz,
                expected_paths=expected_paths,
                dry_run=not self.submit_simulations,
            )

            if not new_Tx:
                self.log("[AL] No new points selected; skipping submit/wait/preprocess for this iteration.")
                continue

            if not self.submit_simulations:
                self.log("[AL] Dry-run: not submitting simulations, not waiting, not updating dataset.")
                continue

            self.log("[AL] Submitting new simulations to HPC...")
            submission_records = self.sim_manager.submit_all_simulations(
                params,
                last_param_name="Tz",
                pairs=True,
                sbatch=True
            )
            self.upsert_registry_rows(self.registry_rows_for_points(
                new_Tx,
                new_Tz,
                expected_paths,
                status="submitted",
                iteration=it + 1,
                submission_records=submission_records,
            ))

            # 6. Wait
            if not self.wait_for_submission_records(submission_records):
                raise TimeoutError("Timed out waiting for MicroLab task completion.")

            if not self.wait_for_simulations(expected_paths):
                raise TimeoutError("Timed out waiting for active-learning simulations.")

            # 7. Preprocess results
            self.log("[AL] Preprocessing new results...")
            from postprocess.preprocess import preprocess_simulation

            done_registry_rows = []
            for zarr_path in expected_paths:
                field, (Tx, Tz), metadata = preprocess_simulation(zarr_path)
                field_key = str(len(dataset.df))
                registry_row = make_registry_row(
                    params={"Tx_val": Tx, "Tz_val": Tz},
                    prefix=self.sim_manager.prefix,
                    split="train",
                    source="active_learning",
                    status="done",
                    iteration=it + 1,
                    param_names=self.param_columns,
                    zarr_path=zarr_path,
                    field_path=self.fields_path,
                    field_key=field_key,
                    extra={
                        key: metadata[key]
                        for key in [
                            "Ty_val",
                            "Aex",
                            "Msat",
                            "Nx",
                            "Ny",
                            "Nz",
                            "dx",
                            "dy",
                            "dz",
                            "target_Nx",
                            "target_Ny",
                            "target_Nz",
                        ]
                        if key in metadata and metadata[key] is not None
                    },
                )
                dataset.add_sample(field, Tx, Tz, {
                    **metadata,
                    "simulation_id": registry_row["simulation_id"],
                    "param_hash": registry_row["param_hash"],
                    "split": "train",
                    "field_key": field_key,
                })
                done_registry_rows.append(registry_row)

            dataset.save(self.meta_path, self.fields_path, normalizer_path=self.normalizer_path)
            self.upsert_registry_rows(done_registry_rows)
            self.log("[AL] Dataset updated.")

        self.log("\n[AL] Active Learning completed successfully.")

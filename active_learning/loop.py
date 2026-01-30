from __future__ import annotations

import os
import time
import shutil
import glob
import numpy as np

from .dataset import MagnetizationDataset
from .trainer import train_cvae
from .model import UNetCVAE
from .uncertainty import compute_uncertainty_map
from .acquisition import select_top_k
from .evaluation import evaluate_model, binned_metrics, save_eval_logs

from .visualization import save_reconstruction_images
from .phase_diagram import predict_phase_on_grid, plot_phase_diagram


# --- importy odporne na to czy uruchamiasz z poziomu repo czy python -m SpinGenix...
try:
    from simulations.swapper import SimulationManager
except ImportError:
    from ..simulations.swapper import SimulationManager  # type: ignore

try:
    from postprocess.preprocess import preprocess_simulation
except ImportError:
    from ..postprocess.preprocess import preprocess_simulation  # type: ignore


class ActiveLearningLoop:
    def __init__(
        self,
        # PATHS
        meta_path="data/dataset/meta.h5",
        fields_path="data/dataset/fields.npz",
        results_dir="results/",
        simulations_dir="/mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/",
        sim_prefix="vxAL",

        # AL PARAMS
        Tx_range=(10e-9, 100e-9),
        Tz_range=(10e-9, 100e-9),
        grid_points=40,
        k_new=20,
        poll_interval=300,
        max_wait_hours=24,
        mc_samples=10,
        require_done=True,  # sprawdzaj *.mx3_status.done (ale z fallbackiem na .zarr)

        # TRAINING PARAMS
        epochs=20,
        batch_size=8,
        lr=1e-4,
        device="cuda",

        # OUTPUTS
        save_visualizations=True,
        save_phase_diagram=True,
        viz_max_samples=16,
        viz_mode="hsl",
        phase_latent_samples=8,

        # SUBMISSION
        use_sbatch=True,
        missing_submit_policy="dry_run",  # "dry_run" lub "error"
    ):
        self.meta_path = meta_path
        self.fields_path = fields_path
        self.results_dir = results_dir

        self.simulations_dir = simulations_dir
        self.sim_prefix = sim_prefix

        self.Tx_range = Tx_range
        self.Tz_range = Tz_range
        self.grid_points = grid_points
        self.k_new = k_new
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours
        self.mc_samples = mc_samples
        self.require_done = require_done

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        self.save_visualizations = save_visualizations
        self.save_phase_diagram = save_phase_diagram
        self.viz_max_samples = viz_max_samples
        self.viz_mode = viz_mode
        self.phase_latent_samples = phase_latent_samples

        self.use_sbatch = use_sbatch
        self.missing_submit_policy = missing_submit_policy

        # paramy które idą do modelu/datasetu
        self.param_cols = ["Tx_val", "Tz_val"]
        self.param_ranges = {"Tx_val": self.Tx_range, "Tz_val": self.Tz_range}

        # Simulation manager (Swapper)
        self.sim_manager = SimulationManager(
            main_path=self.simulations_dir,
            destination_path=self.simulations_dir,
            prefix=self.sim_prefix,
        )

        # foldery wyników
        os.makedirs(self.results_dir, exist_ok=True)
        for sub in ["logs", "reconstructions", "phase_diagrams", "uncertainty_maps"]:
            os.makedirs(os.path.join(self.results_dir, sub), exist_ok=True)

    # ---------------------------------------------------------------
    def _make_dataset(self, split: str | None, return_meta: bool):
        return MagnetizationDataset(
            meta_path=self.meta_path,
            fields_path=self.fields_path,
            split=split,
            param_cols=self.param_cols,
            param_ranges=self.param_ranges,
            target_size=(200, 200),
            return_meta=return_meta,
        )

    def _fmt(self, x: float) -> str:
        # MUSI być zgodne z swapper.py -> format(val, ".5g")
        return format(float(x), ".5g")

    def generate_grid(self):
        Tx_grid = np.linspace(self.Tx_range[0], self.Tx_range[1], self.grid_points)
        Tz_grid = np.linspace(self.Tz_range[0], self.Tz_range[1], self.grid_points)
        return Tx_grid, Tz_grid

    # ---------------------------------------------------------------
    def wait_for_simulations(self, requested_pairs):
        """
        Zwraca dict:
          {(Tx, Tz): zarr_path}
        dla tych, które są gotowe.
        Warunek gotowości:
          - jeśli require_done=True: preferuje *.mx3_status.done,
            ale ma fallback na samo istnienie .zarr (czasem statusy nie powstają).
          - jeśli require_done=False: wystarcza istnienie .zarr.
        """
        start = time.time()
        max_wait = self.max_wait_hours * 3600

        expected = []
        for tx, tz in requested_pairs:
            txs = self._fmt(tx)
            tzs = self._fmt(tz)
            base = os.path.join(self.simulations_dir, self.sim_prefix, f"Tx_{txs}")
            done_path = os.path.join(base, f"Tz_{tzs}.mx3_status.done")
            zarr_path = os.path.join(base, f"Tz_{tzs}.zarr")
            expected.append((float(tx), float(tz), done_path, zarr_path))

        n = len(expected)
        print(f"[AL] Waiting for {n} simulation results...")

        while True:
            matched = {}

            ready_done = 0
            ready_zarr = 0

            for tx, tz, done_path, zarr_path in expected:
                done_ok = os.path.exists(done_path)
                zarr_ok = os.path.exists(zarr_path)

                if done_ok:
                    ready_done += 1
                if zarr_ok:
                    ready_zarr += 1

                if self.require_done:
                    # preferuj DONE, ale fallback na .zarr
                    if done_ok or zarr_ok:
                        matched[(tx, tz)] = zarr_path
                else:
                    if zarr_ok:
                        matched[(tx, tz)] = zarr_path

            if len(matched) == n:
                print("[AL] All simulations ready.")
                return matched

            elapsed = time.time() - start
            print(f"[AL] ready {len(matched)}/{n} (done {ready_done}/{n}, zarr {ready_zarr}/{n}) ... sleeping {self.poll_interval}s.")

            if elapsed > max_wait:
                print("[AL] WARNING: Max waiting time exceeded -> returning partial results.")
                return matched

            time.sleep(self.poll_interval)

    # ---------------------------------------------------------------
    def run(self, iterations=5):
        for it in range(iterations):
            iter_id = it + 1
            print("\n" + "=" * 60)
            print(f"ACTIVE LEARNING ITERATION {iter_id}/{iterations}")
            print("=" * 60)

            # 1) Train dataset
            train_ds = self._make_dataset(split="train", return_meta=False)
            print(f"[AL] Loaded TRAIN dataset: {len(train_ds)} samples.")

            # 2) Test dataset (stały)
            test_ds = self._make_dataset(split="test", return_meta=True)
            print(f"[AL] Loaded TEST dataset: {len(test_ds)} samples.")

            # 3) Train model
            model = UNetCVAE(spatial_size=200)
            model = train_cvae(
                model,
                train_ds,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
            )

            # 4) Evaluate
            if len(test_ds) > 0:
                out = evaluate_model(
                    model=model,
                    dataset=test_ds,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                per_sample = out["per_sample"]
                summary = {k: v for k, v in out.items() if k != "per_sample"}
                summary.update({
                    "iteration": iter_id,
                    "n_train": int(len(train_ds)),
                    "n_test": int(len(test_ds)),
                })

                per_region = binned_metrics(
                    per_sample,
                    param_cols=self.param_cols,
                    metric_cols=["mse", "mae", "meanmz_abs_err"],
                    bins=4,
                )

                save_eval_logs(self.results_dir, iter_id, summary, per_sample, per_region)
                print(f"[AL] TEST metrics saved for iter {iter_id}.")
            else:
                print("[AL] WARNING: No TEST samples -> skipping evaluation.")

            # 4b) reconstructions
            if self.save_visualizations and len(test_ds) > 0:
                out_dir = os.path.join(self.results_dir, "reconstructions", f"iter_{iter_id:03d}")
                save_reconstruction_images(
                    model=model,
                    dataset=test_ds,
                    out_dir=out_dir,
                    max_samples=self.viz_max_samples,
                    mode=self.viz_mode,
                    device=self.device,
                    seed=iter_id,
                )

            # 4c) phase diagram
            if self.save_phase_diagram:
                df_pred, _, _ = predict_phase_on_grid(
                    model=model,
                    Tx_range=self.Tx_range,
                    Tz_range=self.Tz_range,
                    grid_points=self.grid_points,
                    device=self.device,
                    latent_samples=self.phase_latent_samples,
                    spatial_size=200,
                    param_ranges=self.param_ranges,
                )
                phase_path = os.path.join(self.results_dir, "phase_diagrams", f"model_phase_iter_{iter_id:03d}.png")
                plot_phase_diagram(df_pred, title=f"Model Phase (iter {iter_id})", save_path=phase_path, show=False)
                print(f"[AL] Saved phase diagram: {phase_path}")

            # 5) Uncertainty map
            Tx_grid, Tz_grid = self.generate_grid()
            print(f"[AL] Computing uncertainty on {self.grid_points}×{self.grid_points} grid...")
            U = compute_uncertainty_map(
                model,
                Tx_grid,
                Tz_grid,
                mc_samples=self.mc_samples,
                device=self.device,
                param_ranges=self.param_ranges,
            )

            u_path = os.path.join(self.results_dir, "uncertainty_maps", f"U_iter_{iter_id:03d}.npy")
            np.save(u_path, U)
            print(f"[AL] Saved uncertainty map: {u_path}")

            # 6) Select new points
            if self.k_new <= 0:
                print("[AL] k_new <= 0 -> skipping acquisition + simulations.")
                continue

            new_Tx, new_Tz = select_top_k(U, Tx_grid, Tz_grid, K=self.k_new)
            new_Tx = [float(x) for x in new_Tx]
            new_Tz = [float(z) for z in new_Tz]
            print(f"[AL] Selected {len(new_Tx)} new points.")

            pts_path = os.path.join(self.results_dir, "logs", f"selected_points_iter_{iter_id:03d}.csv")
            np.savetxt(pts_path, np.c_[new_Tx, new_Tz], delimiter=",", header="Tx,Tz", comments="")
            print(f"[AL] Saved selected points: {pts_path}")

            # 7) Submit simulations
            if not self.use_sbatch:
                print(
                    f"[AL] use_sbatch=False -> nie submituję symulacji.\n"
                    f"[AL] Punkty zapisane w: {pts_path}\n"
                    "[AL] Kończę iterację."
                )
                break

            sbatch_ok = getattr(self.sim_manager, "sbatch_cmd", None) is not None
            if self.use_sbatch and not sbatch_ok:
                msg = (
                    "[AL] ERROR: sbatch nie jest dostępny (na tym nodzie).\n"
                    f"[AL] Punkty zapisane w: {pts_path}\n"
                    "[AL] Uruchom submit na login node (gdzie działa `sbatch`) albo załaduj moduł Slurm.\n"
                    "[AL] Kończę bez wait_for_simulations (żeby nie wisieć)."
                )
                print(msg)
                if self.missing_submit_policy == "error":
                    raise RuntimeError("sbatch not available; cannot submit simulations.")
                break

            print("[AL] Submitting new simulations via sbatch...")
            params = {"Tx": new_Tx, "Tz": new_Tz}
            self.sim_manager.submit_all_simulations(
                params=params,
                last_param_name="Tz",
                pairs=True,
                sbatch=True,
            )

            # 8) Wait + preprocess + add
            requested_pairs = list(zip(new_Tx, new_Tz))
            matched = self.wait_for_simulations(requested_pairs)

            if len(matched) == 0:
                print("[AL] WARNING: no simulations finished -> nothing to add. Stopping.")
                break

            if len(matched) < len(requested_pairs):
                print(f"[AL] WARNING: only {len(matched)}/{len(requested_pairs)} finished in time; adding what we have.")

            ds_full = self._make_dataset(split=None, return_meta=False)

            added = 0
            for (tx, tz), zarr_path in matched.items():
                try:
                    field, (Tx, Tz), metadata = preprocess_simulation(zarr_path)

                    # safety: (1,H,W,3) -> (H,W,3)
                    if isinstance(field, np.ndarray) and field.ndim == 4 and field.shape[0] == 1:
                        field = field[0]

                    md = dict(metadata)
                    md["Tx_val"] = float(Tx)
                    md["Tz_val"] = float(Tz)

                    ds_full.add_sample(field, md, split="train")
                    added += 1
                except Exception as e:
                    print(f"[AL] SKIP {zarr_path}: {e}")

            # ZAPIS datasetu na dysk (defensywnie, bo implementacje mogły się różnić)
            try:
                ds_full.save(self.meta_path, self.fields_path)
            except TypeError:
                ds_full.save()

            print(f"[AL] Added {added}/{len(matched)} new samples to TRAIN and saved dataset.")

        print("\n[AL] Active Learning completed successfully.")
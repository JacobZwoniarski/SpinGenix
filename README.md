# SpinGenix – Active Learning Framework for Micromagnetic Surrogate Models

SpinGenix is an automated research pipeline that replaces expensive MuMax3
micromagnetic simulations with a learned surrogate model (UNet-CVAE) trained by
Active Learning.

The system:
- submits and manages MuMax3 simulations on HPC nodes,
- preprocesses `.zarr` magnetization data into normalized tensors,
- trains a UNet-CVAE on (Tx, Tz) → magnetization mappings,
- computes uncertainty maps via Monte-Carlo dropout,
- selects the next (Tx, Tz) points to simulate,
- iteratively expands the dataset and improves the model.

---

## Project Structure
active_learning/
model.py               # UNet-CVAE architecture (200×200)
trainer.py             # weighted MSE + KL training loop
dataset.py             # dataset loader/manager
uncertainty.py         # MC dropout uncertainty estimation
acquisition.py         # uncertainty → top-K next points
loop.py                # main Active Learning workflow
phase_diagram.py       # phase classification & plotting
visualization.py       # reconstructions, overlays
utils.py               # helpers, logging, normalization

simulations/
multiple_simulations.py
swapper.py             # SimulationManager (Slurm + MuMax3)
template.mx3           # MuMax3 template file
scripts/               # auto-generated Slurm scripts

postprocess/
preprocess.py          # automatic zarr → 200×200×3 tensor converter
helpers.py             # optional utility code

data/
raw/                   # .zarr simulation outputs
processed/             # intermediate numpy/tensors
dataset/               # final dataset used by Active Learning

results/
reconstructions/
phase_diagrams/
uncertainty_maps/
logs/

run_active_learning.py     # entry point for the pipeline
platform_app/              # local web platform for dataset/results/model preview

---

## Requirements

- Python 3.10+
- PyTorch
- numpy, pandas, matplotlib, seaborn
- discretisedfield
- zarr, h5py, pyzfn
- MuMax3 (GPU version)
- Slurm or similar scheduler

---

## Running the Active Learning Pipeline

Dry-run first, without submitting Slurm jobs:

```bash
PYTHONUNBUFFERED=1 .venv_sg/bin/python -u run_active_learning.py \
  --dry-run \
  --iterations 1 \
  --epochs 1 \
  --grid-points 12 \
  --k-new 5 \
  --mc-samples 4 \
  --device cuda \
  --results-dir results/active_learning/dryrun_$(date +%Y%m%d_%H%M%S) \
  --acquisition-min-distance-nm 0
```

Submit real active-learning simulations explicitly:

```bash
PYTHONUNBUFFERED=1 .venv_sg/bin/python -u run_active_learning.py \
  --submit \
  --iterations 1 \
  --epochs 20 \
  --grid-points 40 \
  --k-new 20 \
  --mc-samples 10 \
  --device cuda \
  --poll-interval 120 \
  --max-wait-hours 24 \
  --results-dir results/active_learning/run_$(date +%Y%m%d_%H%M%S) \
  --simulations-dir /mnt/storage_5/scratch/pl0095-01/jakzwo/simulations/ \
  --simulation-prefix vxAL \
  --amumax-bin /mnt/storage_5/scratch/pl0095-01/bin/amumax \
  --cuda-module cuda/12.6.0_560.28.03
```

This performs:
1. load existing dataset (or initialize from scratch),
2. train the UNet-CVAE,
3. compute an uncertainty grid over (Tx, Tz),
4. select top-K uncertain parameter points,
5. submit MuMax3 simulations via SimulationManager,
6. poll the cluster until new `.zarr` outputs appear,
7. preprocess → incorporate into dataset,
8. iterate.

`run_active_learning.py` intentionally requires either `--dry-run` or
`--submit` so an accidental bare command cannot enqueue a large batch. The loop
trains only on `split=train`; validation/test/boundary holdout points are used
as exclusions for acquisition and are not appended back into training.

---

## Local Platform

The first local dashboard is implemented without extra web dependencies:

```bash
cd /mnt/storage_5/scratch/pl0095-01/jakzwo/SpinGenix_remote
.venv_sg/bin/python platform_app/server.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. The platform reads `data/dataset` and `results`,
shows split/state summaries, phase diagrams, active-learning acquisition CSVs,
available checkpoints, and CPU/CUDA predictions for selected `Tx/Tz`.

The V1 demo flow is documented in `docs/demo_v1_plan.md`. The UI is built
around a two-parameter model demo: checkpoint envelope, preset `Tx/Tz` points,
field generation, physics readout, phase-map context, and active-learning
handoff.

---

## Purpose

SpinGenix enables exploration of micromagnetic phase spaces with minimal
simulation cost, using hybrid classical-ML techniques.

---

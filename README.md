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

python run_active_learning.py

This performs:
1. load existing dataset (or initialize from scratch),
2. train the UNet-CVAE,
3. compute an uncertainty grid over (Tx, Tz),
4. select top-K uncertain parameter points,
5. submit MuMax3 simulations via SimulationManager,
6. poll the cluster until new `.zarr` outputs appear,
7. preprocess → incorporate into dataset,
8. iterate.

---

## Purpose

SpinGenix enables exploration of micromagnetic phase spaces with minimal
simulation cost, using hybrid classical-ML techniques.

---
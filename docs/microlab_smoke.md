# MicroLab smoke submission

This command generates one SpinGenX `.mx3` file and submits it to the MicroLab
task queue in project `spingenx` (`project_id=24`).

Do not pass the CLI token as a command-line argument. Export it in the shell or
provide another environment variable through `--microlab-token-env`.

```bash
export MICROLAB_CLI_TOKEN="<token>"

python3 scripts/submit_microlab_smoke.py \
  --tx-nm 10 \
  --tz-nm 20 \
  --preflight-only
```

`--preflight-only` calls MicroLab `GET /cli/info` and prints the active
configuration, including `project_id=24`, without creating a task or consuming
Slurm/GPU resources.

Submit one real smoke task after the preflight succeeds:

```bash
python3 scripts/submit_microlab_smoke.py \
  --tx-nm 10 \
  --tz-nm 20 \
  --no-wait
```

Defaults:

- API: `https://amucontainers.orion.zfns.eu.org/api/v1`
- project: `24`
- simulations dir:
  `/mnt/storage_6/project_data/pl0095-01/mateuszz/microlab/projects/24_spingenx/workspace/scratch/SpinGenx_remote`
- Slurm resources: `24:00:00`, `1 GPU`, `3 CPU`, `12 GB RAM`
- manifest:
  `results/active_learning/submissions/submission_manifest.json`

Use `--wait` to wait for the MicroLab task to finish before the command exits.
Before creating a new task, the client searches MicroLab project `24` for the
deterministic SpinGenX task name. The manifest then prevents duplicate
submissions for the same `prefix + Tx + Tz` parameter hash across local reruns.

To check task statuses later without submitting anything again:

```bash
python3 scripts/microlab_submission_status.py \
  --submission-manifest results/active_learning/submissions/submission_manifest.json \
  --check-zarr
```

This reads task ids from the manifest and calls MicroLab `GET /tasks/{id}`.
It also writes the latest status, Slurm job id, output directory, and check
timestamp back to the manifest, so reruns can resume without duplicate submits.
With `--check-zarr`, completed MicroLab tasks are split into
`completed_valid_zarr` and `completed_invalid_zarr` based on the expected
SpinGenX `.zarr` markers. By default the status command expects
`project_id=24` and fails if MicroLab reports that a manifest task belongs to a
different project.

Full active-learning submissions use the same defaults:

```bash
python3 run_active_learning.py \
  --preflight-only \
  --submission-backend amucontainers

python3 run_active_learning.py \
  --submit \
  --submission-backend amucontainers \
  --iterations 1 \
  --k-new 20 \
  --max-submit 2
```

Use `--project-id 24` explicitly if running in a shared environment where
`SPINGENX_MICROLAB_PROJECT_ID` might be set by another workflow. Keep
`--max-submit` small for the first production batch; it limits how many selected
points are actually submitted in one active-learning iteration.

For the final end-to-end evidence gate, run:

```bash
python3 scripts/microlab_live_gate.py
```

This performs:

1. MicroLab CLI preflight against `GET /cli/info`.
2. One smoke submit to project `24`.
3. Status polling through `GET /tasks/{id}`.
4. `.zarr` marker validation.
5. Evidence JSON at
   `results/active_learning/submissions/microlab_live_gate_evidence.json`.

Use `--no-wait` only when you want to create the task and inspect it manually
later; the full acceptance gate should run without `--no-wait` so the evidence
contains a completed task and a valid `.zarr`.

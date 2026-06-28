import os
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from simulations.swapper import SimulationManager


@pytest.fixture(autouse=True)
def isolate_local_env_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SPINGENX_ENV_FILE", str(tmp_path / "missing.env"))


def test_load_env_file_reads_local_file_without_overriding(monkeypatch, tmp_path):
    from microlab_env import load_env_file

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join([
            "# local MicroLab credentials",
            "export MICROLAB_CLI_TOKEN='from-file'",
            "MICROLAB_API_BASE=https://from-file.example/api/v1",
        ])
    )

    monkeypatch.delenv("MICROLAB_CLI_TOKEN", raising=False)
    monkeypatch.setenv("MICROLAB_API_BASE", "https://already-set.example/api/v1")

    assert load_env_file(env_file)
    assert os.environ["MICROLAB_CLI_TOKEN"] == "from-file"
    assert os.environ["MICROLAB_API_BASE"] == "https://already-set.example/api/v1"

    assert load_env_file(env_file, override=True)
    assert os.environ["MICROLAB_API_BASE"] == "https://from-file.example/api/v1"


def test_read_registry_falls_back_to_csv_when_parquet_engine_missing(monkeypatch, tmp_path):
    import pandas as pd

    from active_learning.registry import read_registry

    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "simulations.parquet").write_text("placeholder")
    (registry_dir / "simulations.csv").write_text(
        "simulation_id,Tx_val,Tz_val\nsim-1,1e-8,2e-8\n"
    )

    def fail_read_parquet(path):
        raise ImportError("Missing optional dependency 'pyarrow'")

    monkeypatch.setattr(pd, "read_parquet", fail_read_parquet)

    registry = read_registry(registry_dir)

    assert registry.loc[0, "simulation_id"] == "sim-1"
    assert registry.loc[0, "Tx_val"] == pytest.approx(1e-8)


def test_microlab_backend_builds_task_payload_from_env_token(monkeypatch, tmp_path):
    from simulations.backends import MicrolabSubmissionBackend, SlurmResources, SimulationArtifact

    captured = {}

    def fake_transport(method, url, headers, body, timeout):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout"] = timeout
        return 201, {"id": 42, "slurm_job_id": "7151898", "output_dir": "/cluster/run.zarr"}

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    artifact = SimulationArtifact(
        name="Tz_2e-08",
        mx3_path=str(tmp_path / "vxAL" / "Tx_1e-08" / "Tz_2e-08.mx3"),
        zarr_path=str(tmp_path / "vxAL" / "Tx_1e-08" / "Tz_2e-08.zarr"),
        prefix="vxAL",
        params={"Tx": np.float64(1e-8), "Tz": np.float64(2e-8)},
        iteration=7,
    )
    backend = MicrolabSubmissionBackend(
        api_base="https://microlab.example/api/v1/",
        project_id=24,
        resources=SlurmResources(num_cpus=3, memory_gb=12),
        transport=fake_transport,
    )

    record = backend.submit(artifact)

    assert record.backend == "amucontainers"
    assert record.task_id == 42
    assert record.slurm_job_id == "7151898"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://microlab.example/api/v1/tasks/"
    assert captured["headers"]["Authorization"] == "Bearer secret-token-from-env"
    assert "secret-token-from-env" not in repr(captured["body"])

    payload = captured["body"]
    assert payload["simulation_file"] == artifact.mx3_path
    assert payload["cluster"] == "SLURM"
    assert payload["queue_target_mode"] == "SLURM"
    assert payload["partition"] == "proxima"
    assert payload["num_cpus"] == 3
    assert payload["memory_gb"] == 12
    assert payload["num_gpus"] == 1
    assert payload["time_limit"] == "24:00:00"
    assert payload["project_id"] == 24
    assert payload["parameters"]["mx3_file_path"] == artifact.mx3_path
    assert payload["parameters"]["amumax_version_id"] == "default"
    assert payload["parameters"]["spingenx"]["iteration"] == 7
    assert payload["parameters"]["spingenx"]["expected_zarr_path"] == artifact.zarr_path
    assert payload["parameters"]["spingenx"]["params"] == {"Tx": 1e-8, "Tz": 2e-8}
    assert payload["parameters"]["spingenx"]["param_hash"].startswith("sha256:")


def test_simulation_manager_uses_submission_backend_without_local_amumax_validation(tmp_path):
    from simulations.backends import DryRunSubmissionBackend

    backend = DryRunSubmissionBackend()
    manager = SimulationManager(
        main_path=str(tmp_path),
        destination_path=str(tmp_path),
        prefix="vxAL",
        amumax_bin="/missing/amumax",
        submission_backend=backend,
    )

    def fail_validation():
        raise AssertionError("local amumax validation should not run for submission backends")

    manager.validate_amumax_binary = fail_validation

    records = manager.submit_all_simulations(
        {"Tx": np.array([1e-8]), "Tz": np.array([2e-8])},
        last_param_name="Tz",
        pairs=True,
        sbatch=True,
    )

    mx3_path = tmp_path / "vxAL" / "Tx_1e-08" / "Tz_2e-08.mx3"
    assert mx3_path.exists()
    assert len(records) == 1
    assert records[0].backend == "dry-run"
    assert records[0].status == "prepared"
    assert records[0].artifact.mx3_path == str(mx3_path)
    assert records[0].artifact.zarr_path == str(mx3_path.with_suffix(".zarr"))


def test_simulation_manager_keeps_record_alignment_when_some_outputs_exist(tmp_path):
    from simulations.backends import DryRunSubmissionBackend

    complete_zarr = tmp_path / "vxAL" / "Tx_1e-08" / "Tz_2e-08.zarr"
    (complete_zarr / "m_relaxed").mkdir(parents=True)
    (complete_zarr / ".zattrs").write_text("{}")
    (complete_zarr / "m_relaxed" / ".zarray").write_text("{}")

    manager = SimulationManager(
        main_path=str(tmp_path),
        destination_path=str(tmp_path),
        prefix="vxAL",
        submission_backend=DryRunSubmissionBackend(),
    )

    records = manager.submit_all_simulations(
        {"Tx": np.array([1e-8, 3e-8]), "Tz": np.array([2e-8, 4e-8])},
        last_param_name="Tz",
        pairs=True,
        sbatch=True,
    )

    assert len(records) == 2
    assert records[0] is None
    assert records[1].artifact.mx3_path.endswith("Tx_3e-08/Tz_4e-08.mx3")


def test_run_active_learning_builds_microlab_backend_from_cli(monkeypatch):
    from run_active_learning import build_submission_backend, parse_args
    from simulations.backends import MicrolabSubmissionBackend

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")

    args = parse_args([
        "--submit",
        "--submission-backend",
        "amucontainers",
        "--microlab-api-base",
        "https://microlab.example/api/v1",
        "--microlab-token-env",
        "MICROLAB_CLI_TOKEN",
        "--project-id",
        "24",
        "--task-num-cpus",
        "3",
        "--task-memory-gb",
        "12",
        "--task-num-gpus",
        "1",
        "--task-time-limit",
        "24:00:00",
    ])

    backend = build_submission_backend(args)

    assert isinstance(backend, MicrolabSubmissionBackend)
    assert backend.api_base == "https://microlab.example/api/v1"
    assert backend.project_id == 24
    assert backend.resources.num_cpus == 3
    assert backend.resources.memory_gb == 12
    assert backend.resources.num_gpus == 1
    assert backend.resources.time_limit == "24:00:00"


def test_run_active_learning_defaults_to_spingenx_project_24(monkeypatch):
    from run_active_learning import build_submission_backend, parse_args

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    monkeypatch.delenv("MICROLAB_API_BASE", raising=False)

    args = parse_args([
        "--submit",
        "--submission-backend",
        "amucontainers",
    ])

    backend = build_submission_backend(args)

    assert backend.api_base == "https://amucontainers.orion.zfns.eu.org/api/v1"
    assert backend.project_id == 24
    assert args.simulations_dir == (
        "/mnt/storage_6/project_data/pl0095-01/mateuszz/microlab/projects/"
        "24_spingenx/workspace/scratch/SpinGenx_remote"
    )
    assert args.submission_manifest == os.path.join(
        args.results_dir,
        "submissions",
        "submission_manifest.json",
    )


def test_run_active_learning_preflight_only_queries_microlab_project_24(monkeypatch):
    from run_active_learning import main

    requests = []

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        return 200, {
            "user": {"username": "mateuszz"},
            "endpoints": {"tasks": "/api/v1/tasks/"},
        }

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    result = main(
        [
            "--preflight-only",
            "--submission-backend",
            "amucontainers",
        ],
        transport=fake_transport,
    )

    assert result["project_id"] == 24
    assert result["api_base"] == "https://amucontainers.orion.zfns.eu.org/api/v1"
    assert result["cli_info"]["user"]["username"] == "mateuszz"
    assert requests == [(
        "GET",
        "https://amucontainers.orion.zfns.eu.org/api/v1/cli/info",
        {
            "Accept": "application/json",
            "Authorization": "Bearer secret-token-from-env",
        },
        {},
    )]
    assert "secret-token-from-env" not in repr(result)


def test_run_active_learning_passes_max_submit_to_loop(monkeypatch, tmp_path):
    import run_active_learning

    captured = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, iterations):
            captured["iterations"] = iterations

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.setitem(
        sys.modules,
        "active_learning.loop",
        types.SimpleNamespace(ActiveLearningLoop=FakeLoop),
    )

    result = run_active_learning.main([
        "--dry-run",
        "--max-submit",
        "2",
        "--results-dir",
        str(tmp_path / "results"),
        "--simulations-dir",
        str(tmp_path / "SpinGenx_remote"),
    ])

    assert result is None
    assert captured["max_submit"] == 2
    assert captured["iterations"] == 1


def test_microlab_backend_waits_for_task_completion(monkeypatch, tmp_path):
    from simulations.backends import MicrolabSubmissionBackend, SimulationArtifact

    manifest_path = tmp_path / "submission_manifest.json"
    requests = []
    get_responses = iter([
        {"id": 42, "status": "RUNNING"},
        {"id": 42, "status": "COMPLETED", "output_dir": "/cluster/run.zarr"},
    ])

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        if method == "POST":
            return 201, {"id": 42, "status": "PENDING"}
        return 200, next(get_responses)

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    backend = MicrolabSubmissionBackend(
        api_base="https://microlab.example/api/v1",
        manifest_path=str(manifest_path),
        transport=fake_transport,
    )
    artifact = SimulationArtifact(
        name="Tz_2e-08",
        mx3_path=str(tmp_path / "Tz_2e-08.mx3"),
        zarr_path=str(tmp_path / "Tz_2e-08.zarr"),
        prefix="vxAL",
        params={"Tx": 1e-8, "Tz": 2e-8},
    )
    record = backend.submit(artifact)

    assert backend.wait_for_records([record], poll_interval=0, max_wait_hours=0.01)

    get_requests = [request for request in requests if request[0] == "GET"]
    assert [request[1] for request in get_requests] == [
        "https://microlab.example/api/v1/tasks/42",
        "https://microlab.example/api/v1/tasks/42",
    ]
    assert all(
        request[2]["Authorization"] == "Bearer secret-token-from-env"
        for request in get_requests
    )
    manifest = json.loads(manifest_path.read_text())
    entry = next(iter(manifest["submissions"].values()))
    assert entry["status"] == "COMPLETED"
    assert entry["output_dir"] == "/cluster/run.zarr"


def test_microlab_backend_preflight_queries_cli_info_for_project_24(monkeypatch):
    from simulations.backends import MicrolabSubmissionBackend

    requests = []

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        return 200, {
            "user": {"username": "mateuszz"},
            "endpoints": {"tasks": "/api/v1/tasks/"},
        }

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    backend = MicrolabSubmissionBackend(
        api_base="https://amucontainers.orion.zfns.eu.org/api/v1",
        project_id=24,
        transport=fake_transport,
    )

    result = backend.preflight()

    assert result["project_id"] == 24
    assert result["api_base"] == "https://amucontainers.orion.zfns.eu.org/api/v1"
    assert result["cli_info"]["user"]["username"] == "mateuszz"
    assert requests == [(
        "GET",
        "https://amucontainers.orion.zfns.eu.org/api/v1/cli/info",
        {
            "Accept": "application/json",
            "Authorization": "Bearer secret-token-from-env",
        },
        {},
    )]
    assert "secret-token-from-env" not in repr(result)


def test_microlab_backend_reuses_manifest_record_without_duplicate_post(monkeypatch, tmp_path):
    from simulations.backends import MicrolabSubmissionBackend, SimulationArtifact

    requests = []

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        if method == "GET":
            return 200, {"items": []}
        return 201, {"id": 42, "status": "PENDING", "slurm_job_id": "7151898"}

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    manifest_path = tmp_path / "submission_manifest.json"
    artifact = SimulationArtifact(
        name="Tz_2e-08",
        mx3_path=str(tmp_path / "Tz_2e-08.mx3"),
        zarr_path=str(tmp_path / "Tz_2e-08.zarr"),
        prefix="vxAL",
        params={"Tx": 1e-8, "Tz": 2e-8},
    )

    first_backend = MicrolabSubmissionBackend(
        api_base="https://microlab.example/api/v1",
        manifest_path=str(manifest_path),
        transport=fake_transport,
    )
    first = first_backend.submit(artifact)

    second_backend = MicrolabSubmissionBackend(
        api_base="https://microlab.example/api/v1",
        manifest_path=str(manifest_path),
        transport=fake_transport,
    )
    second = second_backend.submit(artifact)

    post_requests = [request for request in requests if request[0] == "POST"]
    assert len(post_requests) == 1
    assert first.task_id == 42
    assert second.task_id == 42
    assert second.status == "existing"
    assert "secret-token-from-env" not in manifest_path.read_text()


def test_microlab_backend_reuses_remote_project_task_before_post(monkeypatch, tmp_path):
    from simulations.backends import MicrolabSubmissionBackend, SimulationArtifact

    requests = []
    artifact = SimulationArtifact(
        name="Tz_2e-08",
        mx3_path=str(tmp_path / "Tz_2e-08.mx3"),
        zarr_path=str(tmp_path / "Tz_2e-08.zarr"),
        prefix="vxAL",
        params={"Tx": 1e-8, "Tz": 2e-8},
    )
    existing_task_name = artifact.task_name()

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        if method == "GET":
            return 200, {
                "items": [
                    {
                        "id": 42,
                        "name": existing_task_name,
                        "status": "RUNNING",
                        "slurm_job_id": "7151898",
                        "output_dir": str(tmp_path / "Tz_2e-08.zarr"),
                        "project_id": 24,
                    }
                ],
            }
        raise AssertionError("remote idempotency should not POST duplicate tasks")

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    manifest_path = tmp_path / "submission_manifest.json"
    backend = MicrolabSubmissionBackend(
        api_base="https://microlab.example/api/v1",
        project_id=24,
        manifest_path=str(manifest_path),
        transport=fake_transport,
    )

    record = backend.submit(artifact)

    assert record.status == "existing_remote"
    assert record.task_id == 42
    assert record.slurm_job_id == "7151898"
    assert record.output_dir == str(tmp_path / "Tz_2e-08.zarr")
    assert len(requests) == 1
    method, url, headers, body = requests[0]
    assert method == "GET"
    assert url.startswith("https://microlab.example/api/v1/tasks/?")
    assert "project_id=24" in url
    assert f"search={existing_task_name}" in url
    assert headers["Authorization"] == "Bearer secret-token-from-env"
    assert body == {}

    manifest = json.loads(manifest_path.read_text())
    entry = next(iter(manifest["submissions"].values()))
    assert entry["task_id"] == 42
    assert entry["status"] == "existing_remote"
    assert "secret-token-from-env" not in manifest_path.read_text()


def test_simulation_artifact_hash_depends_on_params_not_workspace_path(tmp_path):
    from simulations.backends import SimulationArtifact

    first = SimulationArtifact(
        name="Tz_2e-08",
        mx3_path=str(tmp_path / "a" / "Tz_2e-08.mx3"),
        zarr_path=str(tmp_path / "a" / "Tz_2e-08.zarr"),
        prefix="vxAL",
        params={"Tx": 1e-8, "Tz": 2e-8},
    )
    second = SimulationArtifact(
        name="Tz_2e-08",
        mx3_path=str(tmp_path / "b" / "Tz_2e-08.mx3"),
        zarr_path=str(tmp_path / "b" / "Tz_2e-08.zarr"),
        prefix="vxAL",
        params={"Tx": 1e-8, "Tz": 2e-8},
    )

    assert first.normalized_param_hash() == second.normalized_param_hash()


def test_submit_microlab_smoke_generates_one_mx3_and_submits_to_project_24(monkeypatch, tmp_path):
    from scripts.submit_microlab_smoke import main

    requests = []

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        return 201, {"id": 42, "status": "PENDING", "slurm_job_id": "7151898"}

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    result = main(
        [
            "--tx-nm",
            "10",
            "--tz-nm",
            "20",
            "--simulations-dir",
            str(tmp_path / "SpinGenx_remote"),
            "--results-dir",
            str(tmp_path / "results"),
            "--simulation-prefix",
            "vxAL-smoke",
            "--no-wait",
        ],
        transport=fake_transport,
    )

    mx3_path = Path(result["mx3_path"])
    assert mx3_path.exists()
    assert result["task_id"] == 42
    assert result["project_id"] == 24
    assert result["backend"] == "amucontainers"
    assert "secret-token-from-env" not in repr(result)

    assert len(requests) == 2
    lookup_method, lookup_url, lookup_headers, lookup_payload = requests[0]
    assert lookup_method == "GET"
    assert "project_id=24" in lookup_url
    assert lookup_headers["Authorization"] == "Bearer secret-token-from-env"
    assert lookup_payload == {}

    method, url, headers, payload = requests[1]
    assert method == "POST"
    assert url == "https://amucontainers.orion.zfns.eu.org/api/v1/tasks/"
    assert headers["Authorization"] == "Bearer secret-token-from-env"
    assert payload["project_id"] == 24
    assert payload["num_cpus"] == 3
    assert payload["memory_gb"] == 12
    assert payload["num_gpus"] == 1
    assert payload["time_limit"] == "24:00:00"
    assert payload["parameters"]["spingenx"]["params"] == {"Tx": 1e-8, "Tz": 2e-8}


def test_submit_microlab_smoke_default_paths_stay_inside_project_24(monkeypatch):
    from run_active_learning import DEFAULT_SPINGENX_REMOTE_ROOT
    from scripts.submit_microlab_smoke import parse_args

    args = parse_args([
        "--tx-nm",
        "10",
        "--tz-nm",
        "20",
    ])

    assert args.project_id == 24
    assert args.simulations_dir == DEFAULT_SPINGENX_REMOTE_ROOT
    assert "/projects/24_spingenx/workspace/scratch/SpinGenx_remote" in args.simulations_dir


def test_submit_microlab_smoke_preflight_only_does_not_submit(monkeypatch, tmp_path):
    from scripts.submit_microlab_smoke import main

    requests = []

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        return 200, {"user": {"username": "mateuszz"}}

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    result = main(
        [
            "--tx-nm",
            "10",
            "--tz-nm",
            "20",
            "--simulations-dir",
            str(tmp_path / "SpinGenx_remote"),
            "--results-dir",
            str(tmp_path / "results"),
            "--preflight-only",
        ],
        transport=fake_transport,
    )

    assert result["project_id"] == 24
    assert result["cli_info"]["user"]["username"] == "mateuszz"
    assert requests == [(
        "GET",
        "https://amucontainers.orion.zfns.eu.org/api/v1/cli/info",
        {
            "Accept": "application/json",
            "Authorization": "Bearer secret-token-from-env",
        },
        {},
    )]
    assert not (tmp_path / "SpinGenx_remote").exists()
    assert "secret-token-from-env" not in repr(result)


def test_submit_microlab_smoke_reports_existing_complete_zarr_without_submit(
    monkeypatch,
    tmp_path,
):
    from scripts.submit_microlab_smoke import main

    simulations_dir = tmp_path / "SpinGenx_remote"
    existing_zarr = (
        simulations_dir
        / "vxAL-smoke"
        / "Tx_1e-08"
        / "Tz_2e-08.zarr"
    )
    (existing_zarr / "m_relaxed").mkdir(parents=True)
    (existing_zarr / ".zattrs").write_text("{}")
    (existing_zarr / "m_relaxed" / ".zarray").write_text("{}")

    def fail_transport(method, url, headers, body, timeout):
        raise AssertionError("existing complete smoke result should not submit")

    monkeypatch.delenv("MICROLAB_CLI_TOKEN", raising=False)
    result = main(
        [
            "--tx-nm",
            "10",
            "--tz-nm",
            "20",
            "--simulations-dir",
            str(simulations_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--no-wait",
        ],
        transport=fail_transport,
    )

    assert result["status"] == "already_complete"
    assert result["task_id"] is None
    assert result["project_id"] == 24
    assert result["zarr_path"] == str(existing_zarr)


def test_microlab_submission_status_reads_manifest_and_queries_tasks(monkeypatch, tmp_path):
    from scripts.microlab_submission_status import main

    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(json.dumps({
        "submissions": {
            "sha256:abc": {
                "backend": "amucontainers",
                "task_id": 42,
                "status": "PENDING",
                "slurm_job_id": "7151898",
                "mx3_path": "/cluster/input.mx3",
                "zarr_path": "/cluster/input.zarr",
                "prefix": "vxAL-smoke",
                "params": {"Tx": 1e-8, "Tz": 2e-8},
            }
        }
    }))
    requests = []

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        return 200, {
            "id": 42,
            "status": "COMPLETED",
            "slurm_job_id": "7151898",
            "output_dir": "/cluster/input.zarr",
            "project_id": 24,
        }

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    result = main(
        [
            "--submission-manifest",
            str(manifest_path),
        ],
        transport=fake_transport,
    )

    assert result["summary"] == {"COMPLETED": 1}
    assert result["submissions"][0]["task_id"] == 42
    assert result["submissions"][0]["status"] == "COMPLETED"
    assert result["submissions"][0]["output_dir"] == "/cluster/input.zarr"
    assert result["submissions"][0]["project_id"] == 24
    assert requests == [(
        "GET",
        "https://amucontainers.orion.zfns.eu.org/api/v1/tasks/42",
        {
            "Accept": "application/json",
            "Authorization": "Bearer secret-token-from-env",
        },
        {},
    )]
    assert "secret-token-from-env" not in repr(result)

    updated_manifest = json.loads(manifest_path.read_text())
    updated_entry = updated_manifest["submissions"]["sha256:abc"]
    assert updated_entry["status"] == "COMPLETED"
    assert updated_entry["output_dir"] == "/cluster/input.zarr"
    assert updated_entry["slurm_job_id"] == "7151898"
    assert updated_entry["project_id"] == 24
    assert "secret-token-from-env" not in manifest_path.read_text()


def test_microlab_submission_status_rejects_wrong_project(monkeypatch, tmp_path):
    from scripts.microlab_submission_status import main

    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(json.dumps({
        "submissions": {
            "sha256:abc": {
                "backend": "amucontainers",
                "task_id": 42,
                "status": "PENDING",
                "mx3_path": "/cluster/input.mx3",
                "zarr_path": "/cluster/input.zarr",
                "prefix": "vxAL-smoke",
                "params": {"Tx": 1e-8, "Tz": 2e-8},
            }
        }
    }))

    def fake_transport(method, url, headers, body, timeout):
        return 200, {
            "id": 42,
            "status": "COMPLETED",
            "project_id": 25,
        }

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    with pytest.raises(RuntimeError, match="expected project_id=24"):
        main(
            [
                "--submission-manifest",
                str(manifest_path),
            ],
            transport=fake_transport,
        )


def test_microlab_submission_status_can_validate_completed_zarr(monkeypatch, tmp_path):
    from scripts.microlab_submission_status import main

    zarr_path = tmp_path / "input.zarr"
    (zarr_path / "m_relaxed").mkdir(parents=True)
    (zarr_path / ".zattrs").write_text("{}")
    (zarr_path / "m_relaxed" / ".zarray").write_text("{}")

    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(json.dumps({
        "submissions": {
            "sha256:abc": {
                "backend": "amucontainers",
                "task_id": 42,
                "status": "PENDING",
                "slurm_job_id": "7151898",
                "mx3_path": str(tmp_path / "input.mx3"),
                "zarr_path": str(zarr_path),
                "prefix": "vxAL-smoke",
                "params": {"Tx": 1e-8, "Tz": 2e-8},
            }
        }
    }))

    def fake_transport(method, url, headers, body, timeout):
        return 200, {
            "id": 42,
            "status": "COMPLETED",
            "slurm_job_id": "7151898",
            "output_dir": str(zarr_path),
        }

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    result = main(
        [
            "--submission-manifest",
            str(manifest_path),
            "--check-zarr",
        ],
        transport=fake_transport,
    )

    assert result["summary"] == {"completed_valid_zarr": 1}
    assert result["submissions"][0]["status"] == "completed_valid_zarr"
    assert result["submissions"][0]["zarr_complete"] is True

    updated_entry = json.loads(manifest_path.read_text())["submissions"]["sha256:abc"]
    assert updated_entry["status"] == "completed_valid_zarr"
    assert updated_entry["zarr_complete"] is True


def test_microlab_live_gate_writes_end_to_end_evidence(monkeypatch, tmp_path):
    from scripts.microlab_live_gate import main

    requests = []

    def fake_transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, body))
        if url.endswith("/cli/info"):
            return 200, {"user": {"username": "mateuszz"}}
        if "/tasks/?" in url:
            return 200, {"items": []}
        if method == "POST" and url.endswith("/tasks/"):
            zarr_path = Path(body["parameters"]["spingenx"]["expected_zarr_path"])
            (zarr_path / "m_relaxed").mkdir(parents=True)
            (zarr_path / ".zattrs").write_text("{}")
            (zarr_path / "m_relaxed" / ".zarray").write_text("{}")
            return 201, {
                "id": 42,
                "status": "PENDING",
                "slurm_job_id": "7151898",
                "output_dir": str(zarr_path),
                "project_id": 24,
            }
        if url.endswith("/tasks/42"):
            return 200, {
                "id": 42,
                "status": "COMPLETED",
                "slurm_job_id": "7151898",
                "output_dir": str(tmp_path / "SpinGenx_remote" / "vxAL-live-gate" / "Tx_1e-08" / "Tz_2e-08.zarr"),
                "project_id": 24,
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setenv("MICROLAB_CLI_TOKEN", "secret-token-from-env")
    evidence_path = tmp_path / "live_gate_evidence.json"

    result = main(
        [
            "--tx-nm",
            "10",
            "--tz-nm",
            "20",
            "--simulations-dir",
            str(tmp_path / "SpinGenx_remote"),
            "--results-dir",
            str(tmp_path / "results"),
            "--simulation-prefix",
            "vxAL-live-gate",
            "--poll-interval",
            "0",
            "--max-wait-hours",
            "0.01",
            "--evidence-path",
            str(evidence_path),
        ],
        transport=fake_transport,
    )

    assert result["project_id"] == 24
    assert result["preflight"]["project_id"] == 24
    assert result["smoke"]["task_id"] == 42
    assert result["status"]["summary"] == {"completed_valid_zarr": 1}
    assert result["status"]["submissions"][0]["project_id"] == 24
    assert result["status"]["submissions"][0]["zarr_complete"] is True
    assert evidence_path.exists()
    assert json.loads(evidence_path.read_text()) == result
    assert "secret-token-from-env" not in repr(result)
    assert "secret-token-from-env" not in evidence_path.read_text()

    request_urls = [request[1] for request in requests]
    assert "https://amucontainers.orion.zfns.eu.org/api/v1/cli/info" in request_urls
    assert "https://amucontainers.orion.zfns.eu.org/api/v1/tasks/" in request_urls
    assert "https://amucontainers.orion.zfns.eu.org/api/v1/tasks/42" in request_urls

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


Transport = Callable[
    [str, str, Mapping[str, str], Mapping[str, Any], float],
    Tuple[int, Mapping[str, Any]],
]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _safe_task_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value[:120] or "spingenx_task"


@dataclass(frozen=True)
class SlurmResources:
    cluster: str = "SLURM"
    queue_target_mode: str = "SLURM"
    partition: str = "proxima"
    num_cpus: int = 3
    memory_gb: int = 12
    num_gpus: int = 1
    time_limit: str = "24:00:00"
    priority: int = 0
    amumax_version_id: str = "default"


@dataclass(frozen=True)
class SimulationArtifact:
    name: str
    mx3_path: str
    zarr_path: str
    prefix: str
    params: Mapping[str, Any]
    iteration: Optional[int] = None
    run_id: Optional[str] = None
    param_hash: Optional[str] = None

    def normalized_param_hash(self) -> str:
        if self.param_hash:
            return self.param_hash
        return _stable_hash(
            {
                "prefix": self.prefix,
                "params": self.params,
            }
        )

    def task_name(self) -> str:
        parts = ["spingenx", self.prefix, self.name, self.normalized_param_hash()[7:19]]
        return _safe_task_name("_".join(str(part) for part in parts if part))

    def spingenx_metadata(self) -> Dict[str, Any]:
        metadata = {
            "prefix": self.prefix,
            "params": _json_safe(dict(self.params)),
            "param_hash": self.normalized_param_hash(),
            "expected_zarr_path": self.zarr_path,
        }
        if self.iteration is not None:
            metadata["iteration"] = int(self.iteration)
        if self.run_id:
            metadata["run_id"] = self.run_id
        return metadata


@dataclass(frozen=True)
class SubmissionRecord:
    backend: str
    status: str
    artifact: SimulationArtifact
    task_id: Optional[Any] = None
    slurm_job_id: Optional[str] = None
    output_dir: Optional[str] = None
    raw_response: Mapping[str, Any] = field(default_factory=dict)


class MicrolabSubmissionError(RuntimeError):
    pass


class DryRunSubmissionBackend:
    name = "dry-run"

    def submit(self, artifact: SimulationArtifact) -> SubmissionRecord:
        return SubmissionRecord(
            backend=self.name,
            status="prepared",
            artifact=artifact,
        )


class MicrolabSubmissionBackend:
    name = "amucontainers"
    SUCCESS_STATUSES = {"COMPLETED", "DONE", "SUCCESS", "FINISHED"}
    FAILURE_STATUSES = {"FAILED", "CANCELLED", "CANCELED", "ERROR", "TIMEOUT", "NODE_FAIL"}

    def __init__(
        self,
        api_base: str,
        *,
        token: Optional[str] = None,
        token_env: str = "MICROLAB_CLI_TOKEN",
        project_id: Optional[int] = None,
        resources: Optional[SlurmResources] = None,
        timeout: float = 30.0,
        manifest_path: Optional[str] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.token = token if token is not None else os.environ.get(token_env)
        self.token_env = token_env
        self.project_id = project_id
        self.resources = resources or SlurmResources()
        self.timeout = timeout
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.transport = transport or self._default_transport

        if not self.api_base:
            raise ValueError("api_base is required for MicroLab submissions")

    def endpoint(self, path: str) -> str:
        return f"{self.api_base}/{path.lstrip('/')}"

    def headers(self, *, json_body: bool = False) -> Dict[str, str]:
        if not self.token:
            raise ValueError(f"MicroLab CLI token missing; set {self.token_env}")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def preflight(self) -> Mapping[str, Any]:
        status_code, response = self.transport(
            "GET",
            self.endpoint("/cli/info"),
            self.headers(),
            {},
            self.timeout,
        )
        if status_code < 200 or status_code >= 300:
            raise MicrolabSubmissionError(
                f"MicroLab CLI preflight failed with HTTP {status_code}"
            )
        return {
            "api_base": self.api_base,
            "project_id": self.project_id,
            "cli_info": dict(response),
        }

    def build_task_payload(self, artifact: SimulationArtifact) -> Dict[str, Any]:
        resources = self.resources
        payload: Dict[str, Any] = {
            "name": artifact.task_name(),
            "simulation_file": artifact.mx3_path,
            "cluster": resources.cluster,
            "queue_target_mode": resources.queue_target_mode,
            "partition": resources.partition,
            "num_cpus": resources.num_cpus,
            "memory_gb": resources.memory_gb,
            "num_gpus": resources.num_gpus,
            "time_limit": resources.time_limit,
            "priority": resources.priority,
            "auto_start": True,
            "parameters": {
                "submission_method": "spingenx_cli",
                "mx3_file_path": artifact.mx3_path,
                "amumax_version_id": resources.amumax_version_id,
                "spingenx": artifact.spingenx_metadata(),
            },
        }
        if self.project_id is not None:
            payload["project_id"] = self.project_id
        return payload

    def submit(self, artifact: SimulationArtifact) -> SubmissionRecord:
        existing = self._manifest_lookup(artifact)
        if existing is not None:
            return existing

        existing_remote = self.find_existing_task(artifact)
        if existing_remote is not None:
            self._manifest_store(existing_remote)
            return existing_remote

        payload = self.build_task_payload(artifact)
        status_code, response = self.transport(
            "POST",
            self.endpoint("/tasks/"),
            self.headers(json_body=True),
            payload,
            self.timeout,
        )
        if status_code < 200 or status_code >= 300:
            raise MicrolabSubmissionError(
                f"MicroLab task submission failed with HTTP {status_code}"
            )
        task_id = response.get("id", response.get("task_id"))
        slurm_job_id = response.get("slurm_job_id")
        if slurm_job_id is not None:
            slurm_job_id = str(slurm_job_id)
        record = SubmissionRecord(
            backend=self.name,
            status=str(response.get("status", "submitted")),
            artifact=artifact,
            task_id=task_id,
            slurm_job_id=slurm_job_id,
            output_dir=response.get("output_dir"),
            raw_response=dict(response),
        )
        self._manifest_store(record)
        return record

    def find_existing_task(self, artifact: SimulationArtifact) -> Optional[SubmissionRecord]:
        if self.project_id is None:
            return None

        query = urllib.parse.urlencode(
            {
                "project_id": self.project_id,
                "search": artifact.task_name(),
                "cluster": self.resources.cluster,
                "page_size": 10,
            }
        )
        status_code, response = self.transport(
            "GET",
            self.endpoint(f"/tasks/?{query}"),
            self.headers(),
            {},
            self.timeout,
        )
        if status_code < 200 or status_code >= 300:
            raise MicrolabSubmissionError(
                f"MicroLab task lookup failed with HTTP {status_code}"
            )

        items = response.get("items", [])
        if not isinstance(items, list):
            return None
        expected_name = artifact.task_name()
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if item.get("name") != expected_name:
                continue
            task_id = item.get("id", item.get("task_id"))
            slurm_job_id = item.get("slurm_job_id")
            if slurm_job_id is not None:
                slurm_job_id = str(slurm_job_id)
            return SubmissionRecord(
                backend=self.name,
                status="existing_remote",
                artifact=artifact,
                task_id=task_id,
                slurm_job_id=slurm_job_id,
                output_dir=item.get("output_dir"),
                raw_response=dict(item),
            )
        return None

    def _read_manifest(self) -> Dict[str, Any]:
        if self.manifest_path is None or not self.manifest_path.exists():
            return {"submissions": {}}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {"submissions": {}}
        submissions = data.get("submissions")
        if not isinstance(submissions, dict):
            data["submissions"] = {}
        return data

    def _write_manifest(self, data: Mapping[str, Any]) -> None:
        if self.manifest_path is None:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifest_path.with_suffix(f"{self.manifest_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(data), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, self.manifest_path)

    def _manifest_lookup(self, artifact: SimulationArtifact) -> Optional[SubmissionRecord]:
        if self.manifest_path is None:
            return None
        param_hash = artifact.normalized_param_hash()
        data = self._read_manifest()
        entry = data.get("submissions", {}).get(param_hash)
        if not isinstance(entry, dict):
            return None
        return SubmissionRecord(
            backend=str(entry.get("backend", self.name)),
            status="existing",
            artifact=artifact,
            task_id=entry.get("task_id"),
            slurm_job_id=entry.get("slurm_job_id") or None,
            output_dir=entry.get("output_dir") or None,
            raw_response=dict(entry.get("raw_response") or {}),
        )

    def _manifest_store(self, record: SubmissionRecord) -> None:
        if self.manifest_path is None:
            return
        param_hash = record.artifact.normalized_param_hash()
        data = self._read_manifest()
        submissions = data.setdefault("submissions", {})
        submissions[param_hash] = {
            "backend": record.backend,
            "status": record.status,
            "task_id": record.task_id,
            "slurm_job_id": record.slurm_job_id or "",
            "output_dir": record.output_dir or "",
            "mx3_path": record.artifact.mx3_path,
            "zarr_path": record.artifact.zarr_path,
            "prefix": record.artifact.prefix,
            "params": dict(record.artifact.params),
            "raw_response": dict(record.raw_response),
            "updated_at": int(time.time()),
        }
        self._write_manifest(data)

    def _manifest_update_task(self, record: SubmissionRecord, task: Mapping[str, Any]) -> None:
        if self.manifest_path is None:
            return
        param_hash = record.artifact.normalized_param_hash()
        data = self._read_manifest()
        submissions = data.setdefault("submissions", {})
        entry = submissions.get(param_hash)
        if not isinstance(entry, dict):
            entry = {}
            submissions[param_hash] = entry
        entry.update({
            "backend": record.backend,
            "status": str(task.get("status", entry.get("status", ""))),
            "task_id": record.task_id,
            "slurm_job_id": str(task.get("slurm_job_id") or entry.get("slurm_job_id") or ""),
            "output_dir": task.get("output_dir") or entry.get("output_dir") or "",
            "mx3_path": record.artifact.mx3_path,
            "zarr_path": record.artifact.zarr_path,
            "prefix": record.artifact.prefix,
            "params": dict(record.artifact.params),
            "last_checked_at": int(time.time()),
        })
        self._write_manifest(data)

    def get_task(self, task_id: Any) -> Mapping[str, Any]:
        status_code, response = self.transport(
            "GET",
            self.endpoint(f"/tasks/{task_id}"),
            self.headers(),
            {},
            self.timeout,
        )
        if status_code < 200 or status_code >= 300:
            raise MicrolabSubmissionError(
                f"MicroLab task status request failed with HTTP {status_code}"
            )
        return response

    def wait_for_records(
        self,
        records,
        *,
        poll_interval: float = 300.0,
        max_wait_hours: float = 24.0,
    ) -> bool:
        pending = {
            record.task_id: record
            for record in records
            if record.task_id is not None
        }
        if not pending:
            return True

        deadline = time.monotonic() + max_wait_hours * 3600.0
        while pending:
            completed = []
            for task_id in list(pending.keys()):
                task = self.get_task(task_id)
                self._manifest_update_task(pending[task_id], task)
                status = str(task.get("status", "")).upper()
                if status in self.SUCCESS_STATUSES:
                    completed.append(task_id)
                elif status in self.FAILURE_STATUSES:
                    raise MicrolabSubmissionError(
                        f"MicroLab task {task_id} failed with status {status}"
                    )

            for task_id in completed:
                pending.pop(task_id, None)
            if not pending:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval)
        return True

    @staticmethod
    def _default_transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> Tuple[int, Mapping[str, Any]]:
        method = method.upper()
        encoded = None
        if method not in {"GET", "HEAD"}:
            encoded = json.dumps(_json_safe(body)).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=encoded,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw) if raw else {}
                return response.getcode(), payload
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"detail": raw}
            return exc.code, payload


def create_submission_backend(
    submission_backend: str,
    *,
    microlab_api_base: Optional[str] = None,
    microlab_token_env: str = "MICROLAB_CLI_TOKEN",
    project_id: Optional[int] = None,
    resources: Optional[SlurmResources] = None,
    manifest_path: Optional[str] = None,
    transport: Optional[Transport] = None,
) -> Optional[Any]:
    backend = (submission_backend or "local-sbatch").strip().lower()
    if backend in {"local", "local-sbatch", "sbatch"}:
        return None
    if backend in {"dry-run", "dryrun"}:
        return DryRunSubmissionBackend()
    if backend in {"amucontainers", "amu-containers", "microlab"}:
        if not microlab_api_base:
            raise ValueError("--microlab-api-base is required for amucontainers backend")
        return MicrolabSubmissionBackend(
            microlab_api_base,
            token_env=microlab_token_env,
            project_id=project_id,
            resources=resources,
            manifest_path=manifest_path,
            transport=transport,
        )
    raise ValueError(f"Unknown submission backend: {submission_backend}")

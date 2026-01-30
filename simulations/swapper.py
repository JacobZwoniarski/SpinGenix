import os
import filecmp
import errno
import numpy as np
import itertools
import subprocess
import re
import zarr
import logging
import shutil
from typing import Optional, Dict, List, Union
import time

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")


def upgrade_log_level(current_level: str, new_level: str) -> str:
    """
    Promuje poziom logowania.
    Priorytet: ERROR > WARNING > INFO
    """
    levels = ["INFO", "WARNING", "ERROR"]
    return levels[max(levels.index(current_level), levels.index(new_level))]


class SimulationManager:
    def __init__(
        self,
        main_path: str,
        destination_path: str,
        prefix: str,
        template_path: str = "/mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/template.mx3",
        sbatch_cmd: Optional[str] = None,
        amumax_cmd: str = "/mnt/storage_3/home/jakzwo/pl0095-01/scratch/bin/amumax",
    ) -> None:
        self.main_path = main_path
        self.destination_path = destination_path
        self.prefix = prefix
        self.template_path = template_path
        self.amumax_cmd = amumax_cmd

        # sbatch: PATH albo typowe lokalizacje
        self.sbatch_cmd = (
            sbatch_cmd
            or shutil.which("sbatch")
            or ("/usr/bin/sbatch" if os.path.exists("/usr/bin/sbatch") else None)
            or ("/bin/sbatch" if os.path.exists("/bin/sbatch") else None)
        )

    @staticmethod
    def create_path_if_not_exists(file_path: str) -> None:
        """Ensure the directory for a given file path exists."""
        dirn = os.path.dirname(file_path)
        if dirn and not os.path.exists(dirn):
            try:
                os.makedirs(dirn)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    @staticmethod
    def verify_or_replace_file(new_file_path: str, existing_file_path: str) -> bool:
        """Check if a file needs replacing and replace if necessary."""
        if os.path.exists(existing_file_path):
            if filecmp.cmp(new_file_path, existing_file_path, shallow=False):
                os.remove(new_file_path)
                return True
            else:
                os.remove(existing_file_path)
        return False

    @staticmethod
    def find_status_file(path: str, sim_name: str, status: str) -> Optional[str]:
        """Locate a status file based on its name and type."""
        pattern = re.compile(rf"{re.escape(sim_name)}\.mx3_status\.{re.escape(status)}.*")
        if not os.path.isdir(path):
            return None
        for file in os.listdir(path):
            if pattern.match(file):
                return os.path.join(path, file)
        return None

    @staticmethod
    def get_file_status(file_path: str) -> str:
        """Determine the status of a file based on its name."""
        if ".mx3_status.lock" in file_path:
            return "locked"
        elif ".mx3_status.done" in file_path:
            return "finished"
        elif ".mx3_status.interrupted" in file_path:
            return "interrupted"
        return "unknown"

    @staticmethod
    def extract_sim_key(file_path: str) -> str:
        """Extract a concise simulation key from a file path."""
        return os.path.basename(file_path).split(".mx3_status")[0]

    @staticmethod
    def check_simulation_completion(zarr_path: str) -> (bool, int):
        """
        Minimalny warunek do postprocessu w SpinGenix:
        - istnieje dataset 'm_relaxed'
        Fallback:
        - istnieje grupa 'm' (trajektoria) i ma >= 1 frame.
        """
        try:
            z = zarr.open(zarr_path, mode="r")

            if "m_relaxed" in z:
                arr = z["m_relaxed"]
                count = int(arr.shape[0]) if hasattr(arr, "shape") and len(arr.shape) > 0 else 1
                return True, max(count, 1)

            if "m" in z and hasattr(z["m"], "shape"):
                count = int(z["m"].shape[0])
                ok = count >= 1
                return ok, count

            return False, 0
        except Exception:
            return False, 0

    @staticmethod
    def replace_variables_in_template(file_path: str, variables: Dict[str, Union[str, float, int]]) -> str:
        """Replace placeholders in a template file with actual values."""
        with open(file_path, 'r') as file:
            content = file.read()
        for key, value in variables.items():
            content = content.replace(f'{{{key}}}', str(value))
        return content

    def raw_code(self, **kwargs: Dict[str, Union[str, float, int]]) -> str:
        """Generate raw mx3 code from template_path."""
        return SimulationManager.replace_variables_in_template(self.template_path, kwargs)

    def gen_sbatch_script(self, name: str, path: str) -> str:
        """Generate an sbatch script for submitting jobs."""
        mx3_file = f"{path}.mx3"
        lock_file = f"{path}.mx3_status.lock"
        done_file = f"{path}.mx3_status.done"
        interrupted_file = f"{path}.mx3_status.interrupted"

        bad_nodes_file = "/mnt/storage_3/home/jakzwo/bad_nodes.txt"

        exclude_clause = ""
        if os.path.exists(bad_nodes_file):
            with open(bad_nodes_file, 'r') as f:
                nodes = [node.strip() for node in f.readlines() if node.strip()]
            if nodes:
                exclude_clause = f"#SBATCH --exclude={','.join(nodes)}"

        return f"""#!/bin/bash -l
#SBATCH --job-name="{name}"
#SBATCH --mail-type=NONE
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=149GB
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=proxima
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
{exclude_clause}

sleep 10

echo "Running on node: $SLURMD_NODENAME"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

source /mnt/storage_3/home/jakzwo/.bashrc
export TMPDIR="/mnt/storage_3/home/jakzwo/pl0095-01/scratch/tmp/"

if [ ! -f "{bad_nodes_file}" ]; then
    touch "{bad_nodes_file}"
fi

mv "{mx3_file}" "{lock_file}"
"{self.amumax_cmd}" -f --hide-progress-bar -o "{path}.zarr" "{lock_file}"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "FINISHED"
    mv "{lock_file}" "{done_file}"
else
    echo "INTERRUPTED"
    mv "{lock_file}" "{interrupted_file}"

    if grep -q "CUDA_ERROR" "{path}.zarr/amumax.out" || nvidia-smi | grep -q "No devices were found"; then
        if ! grep -q "$SLURMD_NODENAME" "{bad_nodes_file}"; then
            echo "$SLURMD_NODENAME" >> "{bad_nodes_file}"
            echo "Added node $SLURMD_NODENAME to bad nodes list due to CUDA error"
        fi
    fi
fi
"""

    def submit_python_code(
        self,
        code_to_execute: str,
        last_param_name: Optional[str] = None,
        cleanup: bool = False,
        sbatch: bool = True,
        check: bool = False,
        force: bool = False,
        full_name: bool = False,
        **kwargs: Union[str, float, int]
    ) -> None:
        """
        Submit a simulation for given parameters.
        """
        report_lines: List[str] = []
        report_log_level = "INFO"
        restart_required = False
        sim_status_str = "unknown"
        zarr_status_str = "N/A"

        if len(kwargs) > 0 and last_param_name is None:
            last_param_name = list(kwargs.keys())[-1]

        val_sep = "_"
        path = (
            f"{self.main_path}{kwargs['prefix']}/" +
            '/'.join([
                f"{key}{val_sep}{format(val, '.5g') if isinstance(val, (int, float)) else val}"
                for key, val in kwargs.items()
                if key not in [last_param_name, "i", "prefix"]
            ]) + "/"
        )

        # ostatni parametr (np. Tz) robi nazwę pliku
        last_key, last_val = kwargs.popitem()
        sim_name = f"{last_key}{val_sep}{format(last_val, '.5g')}"

        report_lines.append(f"Checking simulation '{sim_name}', PATH: {path}{sim_name}.zarr")

        self.create_path_if_not_exists(path + "dummy.txt")

        lock_file = self.find_status_file(path, sim_name, "lock")
        done_file = self.find_status_file(path, sim_name, "done")
        interrupted_file = self.find_status_file(path, sim_name, "interrupted")

        zarr_path = f"{path}{sim_name}.zarr"
        new_file_path = f"{path}{sim_name}.mx3.tmp"
        existing_file_path = f"{path}{sim_name}.mx3"

        # --- status logic
        if lock_file:
            sim_status_str = "locked"
            zarr_status_str = "not checked"

        elif done_file:
            sim_status_str = "finished"
            ok, cnt = self.check_simulation_completion(zarr_path)
            if ok:
                zarr_status_str = f"complete ({cnt} frames)"
            else:
                zarr_status_str = f"incomplete ({cnt} frames) => restart required"
                restart_required = True
                report_log_level = upgrade_log_level(report_log_level, "ERROR")

        elif interrupted_file:
            sim_status_str = "interrupted => will restart"
            zarr_status_str = "not checked"
            restart_required = True
            try:
                os.remove(interrupted_file)
            except Exception:
                pass

        else:
            if os.path.exists(zarr_path):
                ok, cnt = self.check_simulation_completion(zarr_path)
                if ok:
                    sim_status_str = "done => no status file"
                    zarr_status_str = f"complete ({cnt} frames)"
                else:
                    sim_status_str = "zarr incomplete => restart"
                    zarr_status_str = f"incomplete ({cnt} frames) => restart"
                    restart_required = True
            else:
                sim_status_str = "no status => new => restart"
                zarr_status_str = "no .zarr => start"
                restart_required = True

        # --- restart / submit
        if restart_required:
            with open(new_file_path, "w") as f:
                f.write(code_to_execute)
            os.rename(new_file_path, existing_file_path)

            if sbatch:
                if self.sbatch_cmd is None:
                    report_log_level = upgrade_log_level(report_log_level, "ERROR")
                    sim_status_str = f"{sim_status_str}, but not submitted (sbatch missing)"
                else:
                    sim_sbatch_path = f"{self.main_path}{kwargs['prefix']}/sbatch/{sim_name}.sb"
                    self.create_path_if_not_exists(sim_sbatch_path)
                    with open(sim_sbatch_path, "w") as f:
                        f.write(self.gen_sbatch_script(sim_name, path + sim_name))

                    res = subprocess.run([self.sbatch_cmd, sim_sbatch_path], capture_output=True, text=True)
                    if res.returncode == 0:
                        sim_status_str = f"{sim_status_str} => submitted"
                    else:
                        report_log_level = upgrade_log_level(report_log_level, "ERROR")
                        sim_status_str = f"{sim_status_str} => sbatch failed (rc={res.returncode})"
            else:
                report_log_level = upgrade_log_level(report_log_level, "ERROR")
                sim_status_str = f"{sim_status_str}, but not submitted (sbatch=False)"

        report_lines.append(f"Simulation status: {sim_status_str}")
        report_lines.append(f"Simulation results condition: {zarr_status_str}")

        report_message = "\n".join(report_lines)
        if report_log_level == "ERROR":
            log.error(report_message)
        elif report_log_level == "WARNING":
            log.warning(report_message)
        else:
            log.info(report_message)

    def submit_all_simulations(
        self,
        params: Dict[str, np.ndarray],
        last_param_name: str,
        minsim: int = 0,
        maxsim: Optional[int] = None,
        sbatch: bool = True,
        cleanup: bool = False,
        check: bool = False,
        force: bool = False,
        pairs: bool = False
    ) -> None:
        """
        If pairs=False -> Cartesian product.
        If pairs=True  -> zip(param1[i], param2[i], ...)
        """
        param_names = list(params.keys())

        if pairs:
            param_lengths = [len(params[name]) for name in param_names]
            if len(set(param_lengths)) != 1:
                raise ValueError(
                    "pairs=True requires same length arrays. "
                    f"Found lengths: {dict(zip(param_names, param_lengths))}"
                )
            value_sets = zip(*[params[name] for name in param_names])
        else:
            value_sets = itertools.product(*params.values())

        for i, values in enumerate(value_sets):
            if i < minsim:
                continue
            if maxsim is not None and i >= maxsim:
                break

            kwargs: Dict[str, Union[str, float, int]] = {"prefix": self.prefix, "i": i}
            for name, value in zip(param_names, values):
                kwargs[name] = float(value) if isinstance(value, (int, float, np.floating)) else value

            time.sleep(1)

            self.submit_python_code(
                self.raw_code(**kwargs),
                last_param_name=last_param_name,
                sbatch=sbatch,
                cleanup=cleanup,
                check=check,
                force=force,
                **kwargs
            )
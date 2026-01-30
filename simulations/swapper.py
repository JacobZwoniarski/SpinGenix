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
from typing import Optional, Dict, List, Union, Tuple, Any
import time

from rich.logging import RichHandler

try:
    from .template_manager import TemplateManager
    from .parameter_validator import ParameterValidator
except ImportError:
    from template_manager import TemplateManager
    from parameter_validator import ParameterValidator

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
        prefix: str,
        template_manager: Optional[TemplateManager] = None,
        template_path: Optional[str] = None,
        validator: Optional[ParameterValidator] = None,
        sbatch_cmd: Optional[str] = None,
        amumax_cmd: str = "/mnt/storage_3/home/jakzwo/pl0095-01/scratch/bin/amumax",
        destination_path: Optional[str] = None,  # Deprecated, kept for backward compatibility
    ) -> None:
        self.main_path = main_path
        self.prefix = prefix
        self.amumax_cmd = amumax_cmd
        
        # Template manager setup
        if template_manager is not None:
            self.template_manager = template_manager
        elif template_path is not None:
            self.template_manager = TemplateManager(template_path)
        else:
            # Default fallback
            default_template = "/mnt/storage_2/scratch/pl0095-01/jakzwo/simulations/template.mx3"
            if os.path.exists(default_template):
                self.template_manager = TemplateManager(default_template)
            else:
                # Try relative path
                relative_template = os.path.join(
                    os.path.dirname(__file__), "template.mx3"
                )
                if os.path.exists(relative_template):
                    self.template_manager = TemplateManager(relative_template)
                else:
                    log.warning(
                        "No template provided and default not found. "
                        "Template manager will be unavailable."
                    )
                    self.template_manager = None
        
        # Parameter validator (optional)
        self.validator = validator
        
        # Backward compatibility: keep template_path attribute
        self.template_path = (
            self.template_manager.template_path if self.template_manager else None
        )
        
        # Deprecated parameter warning
        if destination_path is not None:
            log.warning(
                "Parameter 'destination_path' is deprecated and not used. "
                "It will be removed in future versions."
            )
        self.destination_path = destination_path

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
        """
        Generate raw mx3 code from template_path.
        
        Uses TemplateManager if available, falls back to legacy method.
        """
        if self.template_manager is not None:
            return self.template_manager.render(**kwargs)
        elif self.template_path is not None:
            return SimulationManager.replace_variables_in_template(self.template_path, kwargs)
        else:
            raise RuntimeError(
                "No template available. Please provide template_manager or template_path."
            )
    
    def _fmt(self, x: Union[float, int]) -> str:
        """Format numeric value for use in file paths."""
        return format(float(x), ".5g")
    
    def _construct_paths(self, **params) -> Dict[str, str]:
        """
        Construct all file paths for a simulation.
        
        Args:
            **params: Must contain 'prefix', 'last_param_name', and parameter values
            
        Returns:
            Dictionary with keys: base_dir, sim_name, mx3_file, zarr_path,
            sbatch_file, lock_file, done_file, interrupted_file
        """
        prefix = params.get('prefix', self.prefix)
        last_param_name = params['last_param_name']
        last_val = params[last_param_name]
        
        val_sep = "_"
        base_dir = f"{self.main_path}{prefix}/"
        
        # Build path from parameters (excluding last parameter and meta parameters)
        for key, val in params.items():
            if key not in [last_param_name, 'i', 'prefix', 'last_param_name']:
                base_dir += f"{key}{val_sep}{self._fmt(val)}/"
        
        sim_name = f"{last_param_name}{val_sep}{self._fmt(last_val)}"
        
        return {
            'base_dir': base_dir,
            'sim_name': sim_name,
            'mx3_file': os.path.join(base_dir, f"{sim_name}.mx3"),
            'zarr_path': os.path.join(base_dir, f"{sim_name}.zarr"),
            'sbatch_file': os.path.join(self.main_path, prefix, "sbatch", f"{sim_name}.sb"),
            'lock_file': os.path.join(base_dir, f"{sim_name}.mx3_status.lock"),
            'done_file': os.path.join(base_dir, f"{sim_name}.mx3_status.done"),
            'interrupted_file': os.path.join(base_dir, f"{sim_name}.mx3_status.interrupted"),
        }
    
    def _check_simulation_status(self, paths: Dict[str, str]) -> Tuple[str, bool]:
        """
        Check simulation status based on status files and zarr completeness.
        
        Args:
            paths: Dictionary from _construct_paths()
            
        Returns:
            Tuple of (status_string, restart_required)
            
        Status strings:
            - "locked": Simulation currently running
            - "finished": Done file exists and zarr complete
            - "finished_incomplete": Done file exists but zarr incomplete
            - "interrupted": Interrupted file exists
            - "new": No files exist yet
            - "no_status_complete": No done file but zarr is complete
            - "no_status_incomplete": No done file and zarr incomplete
        """
        lock_exists = os.path.exists(paths['lock_file'])
        done_exists = os.path.exists(paths['done_file'])
        interrupted_exists = os.path.exists(paths['interrupted_file'])
        zarr_exists = os.path.exists(paths['zarr_path'])
        
        if lock_exists:
            return "locked", False
        
        if interrupted_exists:
            # Clean up interrupted file
            try:
                os.remove(paths['interrupted_file'])
            except Exception:
                pass
            return "interrupted", True
        
        if done_exists:
            ok, cnt = self.check_simulation_completion(paths['zarr_path'])
            if ok:
                return "finished", False
            else:
                return "finished_incomplete", True
        
        if zarr_exists:
            ok, cnt = self.check_simulation_completion(paths['zarr_path'])
            if ok:
                return "no_status_complete", False
            else:
                return "no_status_incomplete", True
        
        return "new", True
    
    def _write_mx3_file(self, file_path: str, content: str) -> None:
        """
        Write MuMax3 simulation file.
        
        Args:
            file_path: Path to .mx3 file
            content: MuMax3 code content
        """
        self.create_path_if_not_exists(file_path)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = file_path + ".tmp"
        with open(temp_path, 'w') as f:
            f.write(content)
        os.rename(temp_path, file_path)
    
    def _write_sbatch_script(self, file_path: str, content: str) -> None:
        """
        Write SLURM sbatch script.
        
        Args:
            file_path: Path to .sb file
            content: Sbatch script content
        """
        self.create_path_if_not_exists(file_path)
        with open(file_path, 'w') as f:
            f.write(content)
    
    def _submit_to_slurm(self, sbatch_path: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Submit job to SLURM with retry mechanism.
        
        Args:
            sbatch_path: Path to sbatch script
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with keys:
                - success: bool
                - job_id: str or None
                - stdout: str
                - stderr: str
        """
        if self.sbatch_cmd is None:
            return {
                'success': False,
                'job_id': None,
                'stdout': '',
                'stderr': 'sbatch command not available'
            }
        
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    [self.sbatch_cmd, sbatch_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Extract job_id from output: "Submitted batch job 12345"
                    match = re.search(r'Submitted batch job (\d+)', result.stdout)
                    job_id = match.group(1) if match else None
                    
                    return {
                        'success': True,
                        'job_id': job_id,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                else:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        log.warning(
                            f"sbatch failed (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        return {
                            'success': False,
                            'job_id': None,
                            'stdout': result.stdout,
                            'stderr': result.stderr
                        }
            except subprocess.TimeoutExpired:
                if attempt < max_retries - 1:
                    log.warning(
                        f"sbatch timeout (attempt {attempt + 1}/{max_retries}), retrying..."
                    )
                    continue
                else:
                    return {
                        'success': False,
                        'job_id': None,
                        'stdout': '',
                        'stderr': 'Timeout expired'
                    }
            except Exception as e:
                return {
                    'success': False,
                    'job_id': None,
                    'stdout': '',
                    'stderr': str(e)
                }
        
        # Should not reach here, but just in case
        return {
            'success': False,
            'job_id': None,
            'stdout': '',
            'stderr': 'Max retries exceeded'
        }
    
    def submit_single(
        self,
        params: Dict[str, Any],
        force: bool = False,
        sbatch: bool = True
    ) -> Dict[str, Any]:
        """
        Submit a single simulation (new refactored method).
        
        Args:
            params: Parameter dictionary (must contain 'last_param_name' and parameter values)
            force: If True, ignore status and always submit
            sbatch: If True, submit to SLURM; if False, only prepare files
            
        Returns:
            Dictionary with:
                - status: 'submitted' | 'skipped' | 'error'
                - message: Description of what happened
                - paths: Dictionary of file paths
                - job_id: SLURM job ID (if submitted)
        """
        # Validate parameters if validator is available
        if self.validator is not None:
            is_valid, errors = self.validator.validate(params)
            if not is_valid:
                return {
                    'status': 'error',
                    'message': f"Parameter validation failed: {'; '.join(errors)}",
                    'paths': None,
                    'job_id': None
                }
        
        # Construct paths
        try:
            paths = self._construct_paths(**params)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to construct paths: {e}",
                'paths': None,
                'job_id': None
            }
        
        # Check status
        status, restart_required = self._check_simulation_status(paths)
        
        if not restart_required and not force:
            return {
                'status': 'skipped',
                'message': f"Simulation already {status}",
                'paths': paths,
                'job_id': None
            }
        
        # Generate mx3 code
        try:
            mx3_code = self.raw_code(**params)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Template rendering failed: {e}",
                'paths': paths,
                'job_id': None
            }
        
        # Write mx3 file
        try:
            self._write_mx3_file(paths['mx3_file'], mx3_code)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to write mx3 file: {e}",
                'paths': paths,
                'job_id': None
            }
        
        # If sbatch is disabled, stop here
        if not sbatch:
            return {
                'status': 'prepared',
                'message': 'Files prepared but not submitted (sbatch=False)',
                'paths': paths,
                'job_id': None
            }
        
        # Generate and write sbatch script
        try:
            sbatch_content = self.gen_sbatch_script(
                paths['sim_name'],
                os.path.join(paths['base_dir'], paths['sim_name'])
            )
            self._write_sbatch_script(paths['sbatch_file'], sbatch_content)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to write sbatch script: {e}",
                'paths': paths,
                'job_id': None
            }
        
        # Submit to SLURM
        if self.sbatch_cmd is None:
            return {
                'status': 'error',
                'message': 'sbatch command not available',
                'paths': paths,
                'job_id': None
            }
        
        try:
            result = self._submit_to_slurm(paths['sbatch_file'])
            if result['success']:
                return {
                    'status': 'submitted',
                    'message': f"Submitted successfully, job_id={result.get('job_id')}",
                    'paths': paths,
                    'job_id': result.get('job_id')
                }
            else:
                return {
                    'status': 'error',
                    'message': f"Submission failed: {result['stderr']}",
                    'paths': paths,
                    'job_id': None
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Submission exception: {e}",
                'paths': paths,
                'job_id': None
            }
    
    def submit_batch(
        self,
        params: Dict[str, List],
        last_param_name: str,
        pairs: bool = False,
        force: bool = False,
        sbatch: bool = True,
        minsim: int = 0,
        maxsim: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Submit multiple simulations (new refactored method).
        
        Args:
            params: Dictionary mapping parameter names to lists of values
                    Example: {'Tx': [10e-9, 20e-9], 'Tz': [15e-9, 25e-9]}
            last_param_name: Name of the last parameter (used for file naming)
            pairs: If True, use zip (pairs); if False, use cartesian product
            force: If True, ignore status and resubmit all
            sbatch: If True, submit to SLURM
            minsim: Start index for simulations
            maxsim: End index for simulations (None = all)
            
        Returns:
            List of result dictionaries from submit_single()
        """
        param_names = list(params.keys())
        
        # Choose iteration strategy
        if pairs:
            param_lengths = [len(params[name]) for name in param_names]
            if len(set(param_lengths)) != 1:
                raise ValueError(
                    f"pairs=True requires same length arrays. "
                    f"Found lengths: {dict(zip(param_names, param_lengths))}"
                )
            value_sets = list(zip(*[params[name] for name in param_names]))
        else:
            value_sets = list(itertools.product(*params.values()))
        
        results = []
        
        for i, values in enumerate(value_sets):
            if i < minsim:
                continue
            if maxsim is not None and i >= maxsim:
                break
            
            # Build parameter dict for this simulation
            sim_params = {
                'prefix': self.prefix,
                'i': i,
                'last_param_name': last_param_name
            }
            
            for name, value in zip(param_names, values):
                sim_params[name] = float(value) if isinstance(value, (int, float, np.floating)) else value
            
            # Submit single simulation
            result = self.submit_single(sim_params, force=force, sbatch=sbatch)
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0 or result['status'] == 'error':
                log.info(
                    f"[{i + 1}/{len(value_sets)}] {result['status']}: {result['message']}"
                )
            
            # Small delay to avoid overwhelming the scheduler
            if sbatch and result['status'] == 'submitted':
                time.sleep(1)
        
        # Summary
        submitted = sum(1 for r in results if r['status'] == 'submitted')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        errors = sum(1 for r in results if r['status'] == 'error')
        prepared = sum(1 for r in results if r['status'] == 'prepared')
        
        log.info(
            f"Batch complete: {submitted} submitted, {skipped} skipped, "
            f"{errors} errors, {prepared} prepared"
        )
        
        return results

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
        DEPRECATED: Use submit_batch() instead.
        
        Legacy method for backward compatibility.
        If pairs=False -> Cartesian product.
        If pairs=True  -> zip(param1[i], param2[i], ...)
        """
        import warnings
        warnings.warn(
            "submit_all_simulations() is deprecated and will be removed in a future version. "
            "Use submit_batch() instead for better error handling and result tracking.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Delegate to new method
        results = self.submit_batch(
            params=params,
            last_param_name=last_param_name,
            pairs=pairs,
            force=force,
            sbatch=sbatch,
            minsim=minsim,
            maxsim=maxsim
        )
        
        # For backward compatibility, don't return anything
        return None
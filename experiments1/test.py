import os
import shlex
from pathlib import Path

from dotenv import load_dotenv
from hamiltonian import generate_quantum_sk_hamiltonian
from simple_slurm import Slurm


def pt_slurm_wrapper(params, slurm_params):
    load_dotenv()
    scratch_dir = Path(os.getenv("SCRATCH_DIR", "."))
    run_dir = scratch_dir / params["name"]

    # Assemble command-line arguments dynamically
    cmd_args = []
    for key, val in params.items():
        if val is None or val is False:
            continue

        flag = f"--{key}"
        if isinstance(val, bool):
            # Store_true flags are appended without values
            cmd_args.append(flag)
        else:
            # shlex.quote safely escapes Hamiltonian strings containing spaces or symbols
            cmd_args.append(f"{flag} {shlex.quote(str(val))}")

    # Combine into terminal execution command
    exec_cmd = f"python3 -u pt_validation_driver.py {' '.join(cmd_args)}"

    s = Slurm(
        job_name=f"{params["name"]}",
        nodes=f"{slurm_params["slurm-nodes"]}",
        partition=f"{slurm_params["slurm-partition"]}",
        exclusive=True,
        output=f"{scratch_dir / params["name"]}.out",
        time=f"{slurm_params["slurm-time"]}",
    )
    s.add_cmd("source venv/bin/activate")
    s.add_cmd(exec_cmd)
    s.add_cmd(f"mv {scratch_dir / params['name']}.out {run_dir / params['name']}.out")

    s.sbatch()


if __name__ == "__main__":
    name = "static_gamma_0.5"
    h_str, h_gamma_str = generate_quantum_sk_hamiltonian(4, 1.0, 1.5, 0, split=True)

    params = {
        "name": name,
        "h-str": h_str,
        "h-gamma-str": h_gamma_str,
        "beta-min": 0.1,
        "beta-max": 2.0,
        "gamma-min": 0.5,
        "gamma-max": 0.5,
        "n-temperatures": 6,
        "n-procs": 36,
        "n-feedback-loops": 10,
        "measure-speedup": True,
        "validate-final-run": True,
        "test-operator-random-seed": 1001
    }

    slurm_params = {
        "slurm-nodes": 2,
        "slurm-time": "2-00:00:00",
        "slurm-partition": "albash",
    }

    pt_slurm_wrapper(params, slurm_params)

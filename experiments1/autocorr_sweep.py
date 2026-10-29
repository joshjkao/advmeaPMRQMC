import os
import shlex
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from simple_slurm import Slurm


def autocorr_slurm_wrapper(params, slurm_params):
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
    exec_cmd = f"python3 -u single_run_autocorrelation_driver.py {' '.join(cmd_args)}"

    s = Slurm(
        job_name=f"{params['name']}",
        nodes=1,
        ntasks_per_node=1,
        partition=f"{slurm_params['slurm-partition']}",
        output=f"{scratch_dir / params['name']}.out",
        time=f"{slurm_params['slurm-time']}",
    )
    s.add_cmd("source venv/bin/activate")
    s.add_cmd(exec_cmd)
    s.add_cmd(f"mv {scratch_dir / params['name']}.out {run_dir / 'slurm.out'}")

    s.sbatch()


def generate_task_list(n_qubits):
    # Setup your parameters as before
    betas = np.linspace(0.01, 16.0, 10)
    gammas = np.linspace(0.0, 1.0, 5)
    # n_qubits = 16

    tasks = []

    # Generate the exact python commands and save them to a list
    for beta in betas:
        for gamma in gammas:
            for op_idx in range(4):
                for trial_idx in range(8):
                    name = f"sk_autocorrelation_n{n_qubits}/sweep_b{beta:.3f}_g{gamma:.3f}_op{op_idx}_t{trial_idx}"
                    params = {
                        "name": name,
                        "n-qubits": n_qubits,
                        "hamiltonian-random-seed": 42,
                        "beta": float(beta),
                        "gamma": float(gamma),
                        "k-max": 4,
                        "test-operator-random-seed": 1000 + op_idx,
                        "t-steps": 50000,
                        "steps": 1000000,
                    }

                    # Assemble command-line arguments dynamically
                    cmd_args = []
                    for key, val in params.items():
                        if val is None or val is False:
                            continue
                        flag = f"--{key}"
                        if isinstance(val, bool):
                            cmd_args.append(flag)
                        else:
                            cmd_args.append(f"{flag} {shlex.quote(str(val))}")

                    exec_cmd = (
                        f"python3 -u sk_autocorrelation_driver.py {' '.join(cmd_args)}"
                    )
                    tasks.append(exec_cmd)

    return tasks


if __name__ == "__main__":
    n_qubits = 16

    load_dotenv()
    scratch_dir = Path(os.getenv("SCRATCH_DIR", "."))
    task_file = scratch_dir / f"autocorr_sweep_tasks_n{n_qubits}.txt"

    if task_file.exists():
        response = input("task file already exists. redo? (y/n)")
        if response != "y":
            print("aborting")
            sys.exit(1)

    # 1. Write all 1,000,000 commands to a text file
    tasks = generate_task_list(16)
    with open(task_file, "w") as f:
        f.writelines(task + "\n" for task in tasks)

    # 2. Submit a SINGLE job array
    total_jobs = len(tasks)

    s = Slurm(
        job_name=f"sk_sweep_n{n_qubits}",
        nodes=1,
        ntasks_per_node=1,
        partition="albash",
        time="5-00:00:00",
        # %A is the array job ID, %a is the specific task ID
        output=f"{scratch_dir}/logs_n{n_qubits}/sweep_%A_%a.out",
        error=f"{scratch_dir}/logs_n{n_qubits}/sweep_%A_%a.err",
        # This tells SLURM: run tasks 1 through total_jobs, maximum 500 at a time
        array=f"1-{total_jobs}%99",
    )

    s.add_cmd("source venv/bin/activate")
    # Bash magic: read the specific line from the text file corresponding to the array ID
    s.add_cmd(f'CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})')
    # Execute that specific line
    s.add_cmd("$CMD")

    main_array = s.sbatch()
    print(f"Submitted 1 array job to process {total_jobs} tasks.")

    s1 = Slurm(
        job_name=f"process_sk_sweep_n{n_qubits}",
        nodes=1,
        ntasks_per_node=4,
        partition="albash",
        time="00:10:00",
        output=f"{scratch_dir}/logs_n{n_qubits}/process_sweep.out",
        error=f"{scratch_dir}/logs_n{n_qubits}/process_sweep.err",
        dependency={"afterany": main_array},
    )

    s1.add_cmd("source venv/bin/activate")
    s1.add_cmd(
        f"python3 -u process_autocorr_sweep.py --sweep-dir {scratch_dir / f'sk_autocorrelation_n{n_qubits}'} --drop-long-strings"
    )
    s1.sbatch()
    print("Submitted processing script")

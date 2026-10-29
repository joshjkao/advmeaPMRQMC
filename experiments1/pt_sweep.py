import os
import shlex
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from simple_slurm import Slurm


def generate_task_list(n_qubits):
    # Setup your parameters as before
    betas = np.linspace(1.0, 4.0, 4)
    gammas = np.linspace(1.0, 1.0, 1)
    n_temps = [5, 10]

    tasks = []

    # Generate the exact python commands and save them to a list
    for beta in betas:
        for gamma in gammas:
            for n_temp in n_temps:
                for op_idx in range(1):
                    for trial_idx in range(1):
                        name = f"sk_pt_n{n_qubits}/sweep_b{beta:.3f}_g{gamma:.3f}_op{op_idx}_t{trial_idx}_T{n_temp}"
                        params = {
                            "name": name,
                            "n-qubits": n_qubits,
                            "hamiltonian-random-seed": 42,
                            "test-operator-random-seed": 1000 + op_idx,
                            "beta-min": 0.0625,
                            "gamma-min": 0.0,
                            "beta-max": float(beta),
                            "gamma-max": float(gamma),
                            "k-max": 4,
                            "n-temperatures": n_temp,
                            "n-procs": 10,
                            "n-feedback-loops": 10,
                            "measure-speedup": True,
                            "validate-final-run": True,
                            "autocorr-steps": 10000,
                            "t-exchanges-per-iteration": 10000,
                            "exchanges-per-iteration": 10000
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

                        exec_cmd = f"python3 -u sk_pt_driver.py {' '.join(cmd_args)}"
                        tasks.append(exec_cmd)

    return tasks


if __name__ == "__main__":
    n_qubits = 4

    load_dotenv()
    scratch_dir = Path(os.getenv("SCRATCH_DIR", "."))
    task_file = scratch_dir / f"pt_sweep_tasks_n{n_qubits}.txt"

    if task_file.exists():
        response = input("task file already exists. redo? (y/n)")
        if response != "y":
            print("aborting")
            sys.exit(1)

    # 1. Write all 1,000,000 commands to a text file
    tasks = generate_task_list(n_qubits)
    with open(task_file, "w") as f:
        f.writelines(task + "\n" for task in tasks)

    # 2. Submit a SINGLE job array
    total_jobs = len(tasks)

    s = Slurm(
        job_name=f"sk_pt_n{n_qubits}",
        nodes=2,
        exclusive=True,
        partition="albash",
        time="5-00:00:00",
        # %A is the array job ID, %a is the specific task ID
        output=f"{scratch_dir}/pt_logs_n{n_qubits}/sweep_%A_%a.out",
        error=f"{scratch_dir}/pt_logs_n{n_qubits}/sweep_%A_%a.err",
        # This tells SLURM: run tasks 1 through total_jobs, maximum 500 at a time
        array=f"1-{total_jobs}%4",
    )

    s.add_cmd("source venv/bin/activate")
    # Bash magic: read the specific line from the text file corresponding to the array ID
    s.add_cmd(f'CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})')
    # Execute that specific line
    s.add_cmd("$CMD")

    main_array = s.sbatch()
    print(f"Submitted 1 array job to process {total_jobs} tasks.")

    s1 = Slurm(
        job_name=f"process_pt_sweep_n{n_qubits}",
        nodes=1,
        ntasks_per_node=4,
        partition="albash",
        time="00:10:00",
        output=f"{scratch_dir}/pt_logs_n{n_qubits}/process_sweep.out",
        error=f"{scratch_dir}/pt_logs_n{n_qubits}/process_sweep.err",
        dependency={"afterany": main_array},
    )

    s1.add_cmd("source venv/bin/activate")
    s1.add_cmd(
        f"python3 -u process_pt_sweep.py --sweep-dir {scratch_dir / f'sk_autocorrelation_n{n_qubits}'} --drop-long-strings"
    )
    s1.sbatch()
    print("Submitted processing script")

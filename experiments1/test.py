import shlex

from hamiltonian import generate_quantum_sk_hamiltonian
from simple_slurm import Slurm

name = "test_everything_big"
h_str, h_gamma_str = generate_quantum_sk_hamiltonian(8, 1.0, 1.5, 0, split=True)

params = {
    "name": name,
    "h-str": h_str,
    "h-gamma-str": h_gamma_str,
    "beta-min": 0.1,
    "beta-max": 4.0,
    "gamma-min": 1.0,
    "gamma-max": 1.0,
    "n-temperatures": 20,
    "n-procs": 40,
    "n-feedback-loops": 10,
    "measure-speedup": True,
    "validate-final-run": True,
}

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

s = Slurm(nodes=2, partition="albash", exclusive=True, output=f"{params["name"]}.out", time="1-00:00:00")
s.add_cmd("source venv/bin/activate")
s.add_cmd(exec_cmd)
s.add_cmd("make clean")
s.add_cmd(f"mv {params["name"]}.out {params["name"]}")

# Submit job to cluster
s.sbatch()

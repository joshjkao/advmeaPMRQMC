import shutil
import subprocess
import sys
from contextlib import chdir
from pathlib import Path

import numpy as np
from simple_slurm import Slurm
from validation import validate_pt


def write_parameters_hpp(params, output_file):
    Tsteps = params["Tsteps"]
    steps = params["steps"]
    stepsPerMeasurement = params["stepsPerMeasurement"]
    beta = params["beta"]
    tau = params["tau"]
    Nbins = params["Nbins"]
    kmax = params["kmax"]
    save_completed_calculation = params["save_completed_calculation"]
    save_unfinished_calculation = params["save_unfinished_calculation"]
    resume_calculation = params["resume_calculation"]

    with open(output_file, "w") as f:
        f.write(f"""#pragma once
#define Tsteps {Tsteps}
#define steps  {steps}
#define stepsPerMeasurement {stepsPerMeasurement}
#define beta {beta}
#define tau {tau}
#ifndef gamma
#define gamma 1.0 // coefficient multiplying H_gamma for split Hamiltonians
#endif
#define parity_cond 0 // controls parity subspace measurement 

#define MEASURE_HDIAG
#define MEASURE_HDIAG_KINT
static constexpr struct SPECGAP_CONFIG {{
  size_t KMAX{{{kmax}}};
}} specgap_config;
#define TEST_OPERATOR_RANDOM_ZSUM

#define qmax 1000
#define Nbins {Nbins}
#define EXHAUSTIVE_CYCLE_SEARCH
#define GAPS_GEOMETRIC_PARAMETER 0.8
#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9

#define HURRY_ON_SIGTERM\n""")
        if save_completed_calculation:
            f.write("#define SAVE_COMPLETED_CALCULATION\n")
        if save_unfinished_calculation:
            f.write("#define SAVE_UNFINISHED_CALCULATION\n")
        if resume_calculation:
            f.write("#define RESUME_CALCULATION\n")


def write_pt_schedule(betas, output_file):
    with open(output_file, "w") as f:
        f.writelines(f"{beta} {1.0}\n" for beta in betas)


def copy_sources_to_run_directory(run_dir):
    run_dir_path = Path(run_dir)
    sources = [
        "DivDiffExp.hpp",
        "PMRQMC_mpi.cpp",
        "PMRQMC_pt_mpi.cpp",
        "PMRQMC_qcpt_mpi.cpp",
        "Toeplitz.hpp",
        "beta_anneal.hpp",
        "datasummary.cpp",
        "divdiff.hpp",
        "fast_susceptibility.hpp",
        "mainqmc.hpp",
        "multiprog.cpp",
        "prepare.cpp",
        "pt_schedule.hpp",
        "test_operator.hpp",
        "Makefile",
    ]

    for source in sources:
        shutil.copy(source, run_dir_path / source)


def setup_run_dir(run_dir, h_string):
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=False)

    copy_sources_to_run_directory(run_dir)

    with open(run_dir_path / "H.txt", "w") as f:
        f.write(h_string)


def execute_run_native(run_dir, params):
    run_dir_path = Path(run_dir)
    write_parameters_hpp(params, run_dir_path / "parameters.hpp")
    nproc = params["nproc"]
    with chdir(run_dir_path):
        subprocess.run(["make", "PMRQMC_mpi.bin"], check=False)
        with open("PMRQMC_mpi.out", "w") as out_file:
            subprocess.run(
                ["mpirun", "-n", f"{nproc}", "PMRQMC_mpi.bin"],
                stdout=out_file,
                check=False,
            )


def execute_pt_run_native(run_dir, params):
    run_dir_path = Path(run_dir)
    write_parameters_hpp(params, run_dir_path / "parameters.hpp")
    write_pt_schedule(params["beta_range"], run_dir_path / "schedule.txt")
    validate = params["validate"]
    nproc = params["nproc"]
    with chdir(run_dir_path):
        subprocess.run(["make", "PMRQMC_qcpt_mpi.bin"], check=False)
        subprocess.run(
            [
                "mpirun",
                "-n",
                f"{nproc}",
                "PMRQMC_qcpt_mpi.bin",
                "--schedule",
                "schedule.txt",
                "--updates-per-exchange",
                f"{params['updates_per_exchange']}",
            ],
            check=False,
        )
        if validate:
            validation_data = validate_pt(Path("."))
            validation_data.to_csv("validation_out.csv")


def execute_run_slurm(run_dir, params, slurm_params):
    run_dir_path = Path(run_dir)
    write_parameters_hpp(params, run_dir_path / "parameters.hpp")

    nodes = slurm_params["nodes"]
    partition = slurm_params["partition"]
    time = slurm_params["time"]
    job_name = slurm_params["job_name"]

    slurm = Slurm(
        job_name=job_name, nodes=nodes, exclusive=True, partition=partition, time=time
    )
    with chdir(run_dir_path):
        slurm.add_cmd("touch running")
        slurm.add_cmd("make PMRQMC_mpi.bin")
        slurm.add_cmd(
            "srun --mpi=pmix_v2 -o PMRQMC_mpi.out -e slurm-%j.out -n $SLURM_NTASKS ./PMRQMC_mpi.bin"
        )
        slurm.add_cmd("rm running")
        slurm.sbatch()
        print(slurm)


def execute_pt_run_slurm(run_dir, params, slurm_params):
    run_dir_path = Path(run_dir)
    write_parameters_hpp(params, run_dir_path / "parameters.hpp")
    write_pt_schedule(params["beta_range"], run_dir_path / "schedule.txt")

    nodes = slurm_params["nodes"]
    tasks_per_node = slurm_params["tasks_per_node"]
    partition = slurm_params["partition"]
    time = slurm_params["time"]
    job_name = slurm_params["job_name"]
    updates_per_exchange = params["updates_per_exchange"]
    validate = params["validate"]

    slurm = Slurm(
        job_name=job_name, nodes=nodes, exclusive=True, partition=partition, time=time
    )
    with chdir(run_dir_path):
        slurm.add_cmd("touch running")
        slurm.add_cmd("make PMRQMC_qcpt_mpi.bin")
        slurm.add_cmd(
            f"srun --mpi=pmix_v2 -o PMRQMC_qcpt_mpi.out -e slurm-%j.out -n {nodes * tasks_per_node} ./PMRQMC_qcpt_mpi.bin --schedule schedule.txt --updates-per-exchange {updates_per_exchange}"
        )
        if validate:
            slurm.add_cmd("source ~/advmeaPMRQMC/experiments1/.env/bin/activate")
            slurm.add_cmd("python3 ~/advmeaPMRQMC/experiments1/validation.py .")
        slurm.add_cmd("rm running")
        slurm.sbatch()
        print(slurm)


if __name__ == "__main__":
    test_run_dir = Path("test_run_dir")
    test_h = """-1.5 1 X\n -1.5 2 X\n 0.5 1 Z 2 Z\n"""

    test_params = {
        "Tsteps": 1000,
        "steps": 10000,
        "stepsPerMeasurement": 10,
        "beta": 0.0625,
        "beta_range": np.geomspace(0.01, 3.0, 20),
        "tau": 0.0625 / 2,
        "kmax": 4,
        "Nbins": 100,
        "updates_per_exchange": 10,
        "validate": True,
    }

    slurm_params = {
        "job_name": "test",
        "nodes": 1,
        "tasks_per_node": 20,
        "partition": "albash",
        "time": "00:02:00",
    }

    setup_run_dir(test_run_dir, test_h)
    # execute_run_native(test_run_dir, test_params)
    # execute_run_slurm(test_run_dir, test_params, None)

    # execute_pt_run_native(test_run_dir, test_params)
    execute_pt_run_slurm(test_run_dir, test_params, slurm_params)

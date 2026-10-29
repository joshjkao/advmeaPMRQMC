import logging
import shutil
import subprocess
from contextlib import chdir
from pathlib import Path

import numpy as np
from simple_slurm import Slurm
from validation import validate_pt, validate_single_temp

logger = logging.getLogger(__name__)

def write_parameters_hpp(params, output_file):
    Tsteps = params["Tsteps"]
    steps = params["steps"]
    stepsPerMeasurement = params["stepsPerMeasurement"]
    beta = params["beta"]
    gamma = params["gamma"]
    tau = params["tau"]
    Nbins = params["Nbins"]
    kmax = params["kmax"]
    save_completed_calculation = params.get("save_completed_calculation", False)
    save_unfinished_calculation = params.get("save_unfinished_calculation", False)
    resume_calculation = params.get("resume_calculation", False)
    save_timeseries = params.get("save_timeseries", False)
    test_operator_random_seed = params.get("test_operator_random_seed", None)

    lines = [
        "#pragma once",
        f"#define Tsteps {Tsteps}",
        f"#define steps  {steps}",
        f"#define stepsPerMeasurement {stepsPerMeasurement}",
        f"#define beta {beta}",
        f"#define tau {tau}",
        "#ifndef gamma",
        f"#define gamma {gamma} // coefficient multiplying H_gamma for split Hamiltonians",
        "#endif",
        "#define parity_cond 0 // controls parity subspace measurement",
        "",
        "#define MEASURE_HDIAG",
        "#define MEASURE_HDIAG_KINT",
        "static constexpr struct SPECGAP_CONFIG {",
        f"  size_t KMAX{{{kmax}}};",
        "} specgap_config;",
        "#define TEST_OPERATOR_RANDOM_ZSUM",
        "",
        "#define qmax 1000",
        f"#define Nbins {Nbins}",
        "#define EXHAUSTIVE_CYCLE_SEARCH",
        "#define GAPS_GEOMETRIC_PARAMETER 0.8",
        "#define COMPOSITE_UPDATE_BREAK_PROBABILITY 0.9",
        "",
        "#define HURRY_ON_SIGTERM",
    ]

    if save_completed_calculation:
        lines.append("#define SAVE_COMPLETED_CALCULATION")
    if save_unfinished_calculation:
        lines.append("#define SAVE_UNFINISHED_CALCULATION")
    if resume_calculation:
        lines.append("#define RESUME_CALCULATION")
    if save_timeseries:
        lines.append("#define SAVE_TIMESERIES")
    if test_operator_random_seed != None:
        lines.append(f"#define TEST_OPERATOR_RANDOM_SEED {test_operator_random_seed}")

    new_content = "\n".join(lines) + "\n"

    target_path = Path(output_file)

    # Check if the file exists and if the content matches
    if target_path.exists() and target_path.read_text() == new_content:
        return

    target_path.write_text(new_content)
    print(f"Updated {target_path.name}.")


def write_pt_schedule(betas, gammas, output_file):
    with open(output_file, "w") as f:
        f.writelines(f"{beta} {gamma}\n" for beta, gamma in zip(betas, gammas))


def copy_sources_to_run_directory(run_dir):
    run_dir_path = Path(run_dir)
    this_dir = Path(__file__).resolve().parent
    source_dir = this_dir / ".."
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
        shutil.copy(source_dir / source, run_dir_path / source)


def setup_run_dir(run_dir, h_string, h_gamma_string=None):
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=False)

    copy_sources_to_run_directory(run_dir)

    with open(run_dir_path / "H.txt", "w") as f:
        f.write(h_string)

    if h_gamma_string != None:
        with open(run_dir_path / "H_gamma.txt", "w") as f:
            f.write(h_gamma_string)


def execute_run_native(run_dir, params):
    run_dir_path = Path(run_dir)
    validate = params["validate"]

    if "nproc" not in params:
        raise KeyError("Execution failed: 'nproc' is missing from params.")

    try:
        if not run_dir_path.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir_path}")

        write_parameters_hpp(params, run_dir_path / "parameters.hpp")

        with chdir(run_dir_path):
            try:
                subprocess.run(
                    ["make", "PMRQMC_mpi.bin"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Compilation failed! Stderr: {e.stderr.strip()}")
                raise  # crash immediately

            with (
                open("PMRQMC_mpi.out", "w") as out_file,
                open("PMRQMC_mpi.err", "w") as err_file,
            ):
                subprocess.run(
                    ["mpirun", "-n", str(params["nproc"]), "PMRQMC_mpi.bin"],
                    stdout=out_file,
                    stderr=err_file,
                    check=True,
                )
            if validate:
                validation_data, metadata = validate_single_temp(Path("."))
                for key, val in metadata.items():
                    validation_data[key] = val
                validation_data.to_csv("validation_out.csv")
                
            if params.get("cleanup", False):
                subprocess.run(["make", "clean"], check=True)

    except Exception as e:
        logger.error(f"Job in {run_dir} failed: {type(e).__name__} - {e!s}")
        raise


def execute_pt_run_native(run_dir, params):
    run_dir_path = Path(run_dir)
    write_parameters_hpp(params, run_dir_path / "parameters.hpp")
    write_pt_schedule(
        params["beta_range"], params["gamma_range"], run_dir_path / "schedule.txt"
    )
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
        if params.get("cleanup", False):
            subprocess.run(["make", "clean"], check=False)


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

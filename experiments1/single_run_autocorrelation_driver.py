import argparse
import json
import os
from pathlib import Path

from autocorr import calc_autocorr
from dotenv import load_dotenv
from pmrqmc_mpi_driver import (
    execute_run_native,
    setup_run_dir,
)


def single_run_autocorrelation(main_params):

    name = main_params["name"]
    h_str = main_params["h_str"]
    h_gamma_str = main_params["h_gamma_str"]
    beta = main_params["beta"]
    gamma = main_params["gamma"]
    test_operator_random_seed = main_params["test_operator_random_seed"]
    t_steps = main_params["t_steps"]
    steps = main_params["steps"]
    k_max = main_params["k_max"]

    load_dotenv()
    scratch_dir = Path(os.getenv("SCRATCH_DIR", "."))
    run_dir = scratch_dir / name

    # guarantee that the directory exists before anything else happens
    if not run_dir.exists():
        setup_run_dir(run_dir, h_str, h_gamma_str)
    else:
        with open(run_dir / "H.txt", "w") as f:
            f.write(h_str)
        if h_gamma_str != None:
            with open(run_dir / "H_gamma.txt", "w") as f:
                f.write(h_gamma_str)

    autocorr_params = {
        "Tsteps": t_steps,
        "steps": steps,
        "stepsPerMeasurement": 1,
        "beta": beta,
        "gamma": gamma,
        "tau": beta / 2,
        "kmax": k_max,
        "Nbins": steps,
        "validate": False,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "save_timeseries": True,
        "nproc": 1,
        "test_operator_random_seed": test_operator_random_seed,
        "cleanup": True
    }

    print("executing short run for autocorrelation statistics")
    execute_run_native(run_dir, autocorr_params)

    tau, bin_length, ok = calc_autocorr(run_dir / "timeseries_rank0.txt")
    print(f"tau was {tau}")

    instance_details = main_params
    instance_details["tau"] = tau
    instance_details["bin_length"] = bin_length
    instance_details["ok"] = ok

    with open(run_dir / "autocorr.json", "w") as f:
        json.dump(instance_details, f)
    instance_details_str = json.dumps(instance_details, indent=2)
    print(instance_details_str)

def parse_args():
    parser = argparse.ArgumentParser(description="Does a single autocorrelation run")

    # Identifiers & Hamiltonian Strings
    parser.add_argument(
        "--name", type=str, default="pt_run", help="Run name/identifier"
    )
    parser.add_argument(
        "--h-str", type=str, required=True, help="Hamiltonian string representation"
    )
    parser.add_argument(
        "--h-gamma-str",
        type=str,
        default=None,
        help="Gamma Hamiltonian string representation",
    )

    # Range Parameters (Floats)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0625,
        help="inverse temperature (beta)",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="transverse field (gamma)"
    )
    parser.add_argument("--k-max", type=int, required=True, help="maximum k value")

    parser.add_argument(
        "--test-operator-random-seed",
        type=int,
        default=42,
        help="seed to generate the test operator",
    )
    parser.add_argument(
        "--t-steps",
        type=int,
        default=10000,
        help="equilibration steps",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="number of steps",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    params = parse_args()

    single_run_autocorrelation(params)

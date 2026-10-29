import json
import os
from pathlib import Path

import numpy as np
from autocorr import calc_autocorr
from dotenv import load_dotenv
from pmrqmc_mpi_driver import (
    execute_run_native,
    setup_run_dir,
)


def single_temp(main_params):

    name = main_params["name"]
    h_str = main_params["h_str"]
    h_gamma_str = main_params["h_gamma_str"]
    beta = main_params["beta"]
    gamma = main_params["gamma"]
    test_operator_random_seed = main_params["test_operator_random_seed"]
    autocorr_steps = main_params["autocorr_steps"]
    k_max = main_params["k_max"]
    tau_per_bin = main_params["tau_per_bin"]
    target_t_nbins = main_params["target_t_nbins"]
    target_nbins = main_params["target_nbins"]
    tau = main_params["tau"]
    nproc = main_params["nproc"]

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

    instance_details = main_params
    with open(run_dir / "instance_details.json", "w") as f:
        json.dump(instance_details, f)

    if tau == None:
        print("determining tau automatically")

        autocorr_params = {
            "Tsteps": autocorr_steps,
            "steps": autocorr_steps,
            "stepsPerMeasurement": 1,
            "beta": beta,
            "gamma": gamma,
            "tau": beta / 2,
            "kmax": k_max,
            "Nbins": autocorr_steps,
            "validate": False,
            "save_completed_calculation": False,
            "save_unfinished_calculation": False,
            "resume_calculation": False,
            "save_timeseries": True,
            "nproc": 1,
            "test_operator_random_seed": test_operator_random_seed,
            "cleanup": True,
        }

        execute_run_native(run_dir, autocorr_params)

        tau, _bin_length, ok = calc_autocorr(run_dir / "timeseries_rank0.txt")
        instance_details["ok"] = ok
        instance_details["tau_calculated"] = True
        print(f"tau was calculated to be {tau}")
    else:
        print(f"using tau={tau}")
        instance_details["tau_calculated"] = False

    instance_details["tau"] = tau

    tau_corr = int(np.ceil(tau))
    target_bin_length = tau_corr * tau_per_bin

    with open(run_dir / "instance_details.json", "w") as f:
        json.dump(instance_details, f)

    final_params = {
        "Tsteps": target_bin_length * target_t_nbins,
        "steps": target_bin_length * target_nbins,
        "stepsPerMeasurement": tau_corr,
        "beta": beta,
        "gamma": gamma,
        "tau": beta / 2,
        "kmax": k_max,
        "Nbins": target_nbins,
        "validate": False,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "save_timeseries": False,
        "nproc": nproc,
        "test_operator_random_seed": test_operator_random_seed,
        "cleanup": True,
    }

    execute_run_native(run_dir, final_params)

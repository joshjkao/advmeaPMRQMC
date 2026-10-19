import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from autocorr import calc_autocorr
from dotenv import load_dotenv
from hamiltonian import generate_1d_tfi, generate_quantum_sk_hamiltonian, toy_3bit
from pmrqmc_mpi_driver import execute_pt_run_native, execute_run_native, setup_run_dir
from pt_optimize import optimize_schedule, optimize_schedule_interpolated
from roundtrips import parse_rt_stats


def full_validation_run(main_params):

    name = main_params["name"]
    h_str = main_params["h_str"]
    h_gamma_str = main_params["h_gamma_str"]
    beta_min = main_params["beta_min"]
    beta_max = main_params["beta_max"]
    gamma_min = main_params["gamma_min"]
    gamma_max = main_params["gamma_max"]
    n_temperatures = main_params["n_temperatures"]
    n_procs = main_params["n_procs"]
    n_feedback_loops = main_params["n_feedback_loops"]
    measure_speedup = main_params["measure_speedup"]
    validate_final_run = main_params["validate_final_run"]

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
        "Tsteps": 20000,
        "steps": 20000,
        "stepsPerMeasurement": 1,
        "beta": beta_min,
        "gamma": gamma_min,
        "tau": beta_min / 2,
        "kmax": 5,
        "Nbins": 20000,
        "validate": False,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "save_timeseries": True,
        "nproc": 1,
    }

    print("executing short run for autocorrelation statistics")
    execute_run_native(run_dir, autocorr_params)

    tau, bin_length = calc_autocorr(run_dir / "timeseries_rank0.txt")
    print(f"tau was {tau}")

    print("starting PT temperature optimization")
    init_beta_schedule = np.geomspace(beta_max, beta_min, n_temperatures)
    init_gamma_schedule = np.geomspace(gamma_max, gamma_min, n_temperatures)
    tau_corr = int(np.ceil(tau))

    therm_exchanges = 10000
    tuning_exchanges = 10000

    params = {
        "Tsteps": therm_exchanges * tau_corr,
        "steps": tuning_exchanges * tau_corr,
        "stepsPerMeasurement": tau_corr,
        "beta": beta_min,
        "beta_range": init_beta_schedule,
        "gamma_range": init_gamma_schedule,
        "gamma": gamma_min,
        "tau": beta_min / 2,
        "kmax": 5,
        "Nbins": 100,
        "validate": False,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "updates_per_exchange": tau_corr,
        "save_timeseries": False,
        "nproc": n_procs,
    }

    avg_rt_length = None

    for iteration in range(n_feedback_loops):
        print(f"\n--- Running PT Optimization Iteration {iteration + 1} of {n_feedback_loops} ---")
        execute_pt_run_native(run_dir, params)

        total_visits, round_trips = parse_rt_stats(run_dir / "pmrqmc_pt_flow.csv")

        if round_trips > 0:
            avg_rt_length = total_visits / round_trips
            print(
                f"Iteration {iteration} avg RT length: {avg_rt_length:.2f} exchange attempts"
            )
        else:
            print(
                f"Warning: 0 round trips completed in iteration {iteration}. Increase tuning_exchanges."
            )
            avg_rt_length = tuning_exchanges  # Fallback estimate

        _, old_schedule, new_schedule = optimize_schedule(
            run_dir / "pmrqmc_pt_flow.csv",
            run_dir / "schedule.txt",
            run_dir / "schedule.txt",
        )

        # old_betas = old_schedule[:, 0]
        new_betas = new_schedule[:, 0]

        # old_gammas = old_schedule[:, 1]
        new_gammas = new_schedule[:, 1]

        params["beta_range"] = new_betas
        params["gamma_range"] = new_gammas

        if np.allclose(old_schedule, new_schedule, rtol=0.01):
            print("Temperature schedule converged. Stopping optimization early.")
            break

    # Target bin length calculation for final production run
    int_rt_length = int(np.ceil(avg_rt_length))
    target_bin_length = int_rt_length * tau_corr * 100
    print(f"Final target bin length: {target_bin_length}")

    if measure_speedup:
        autocorr_params = {
            "Tsteps": 20000,
            "steps": 20000,
            "stepsPerMeasurement": 1,
            "beta": beta_max,
            "gamma": gamma_max,
            "tau": beta_min / 2,
            "kmax": 5,
            "Nbins": 20000,
            "validate": False,
            "save_completed_calculation": False,
            "save_unfinished_calculation": False,
            "resume_calculation": False,
            "save_timeseries": True,
            "nproc": 1,
        }
        execute_run_native(run_dir, autocorr_params)
        tau_max_beta, _ = calc_autocorr(run_dir / "timeseries_rank0.txt")
        print(f"tau at max beta was {tau_max_beta}")
        print(
            f"the respective timescales are {tau_max_beta} (single temperature) and {int_rt_length * tau_corr} (PT)"
        )
        print(
            f"the relative speedup is {tau_max_beta / (int_rt_length * tau_corr) / n_temperatures}"
        )

    timing_data = pd.read_csv(run_dir / "pmrqmc_pt_rank_timing.csv")
    wall_time = np.max(timing_data["elapsed_seconds"])
    time_per_update = wall_time / (tuning_exchanges) / tau_corr

    therm_target_bins = 10
    final_target_bins = 50
    therm_target_steps = therm_target_bins * target_bin_length
    target_steps = final_target_bins * target_bin_length

    expected_time = time_per_update * target_steps

    final_params = {
        "Tsteps": therm_target_steps,
        "steps": target_steps,
        "stepsPerMeasurement": tau_corr,
        "updates_per_exchange": tau_corr,
        "beta": beta_min,
        "beta_range": params["beta_range"],
        "gamma": gamma_min,
        "gamma_range": params["gamma_range"],
        "tau": beta_min / 2,
        "kmax": 5,
        "Nbins": final_target_bins,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "save_timeseries": False,
        "nproc": n_procs,
        "validate": validate_final_run,
    }

    print(f"starting final run. expected time: {expected_time} seconds")

    execute_pt_run_native(run_dir, final_params)


def parse_args():
    parser = argparse.ArgumentParser(
        description="QMC Parallel Tempering Simulation & Validation Pipeline"
    )

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
        "--beta-min",
        type=float,
        default=0.0625,
        help="Minimum inverse temperature (beta)",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=16.0,
        help="Maximum inverse temperature (beta)",
    )
    parser.add_argument(
        "--gamma-min", type=float, default=0.5, help="Minimum transverse field (gamma)"
    )
    parser.add_argument(
        "--gamma-max", type=float, default=1.5, help="Maximum transverse field (gamma)"
    )

    # Execution Parameters (Integers)
    parser.add_argument(
        "--n-temperatures",
        type=int,
        default=10,
        help="Number of PT replicas/schedule points",
    )
    parser.add_argument(
        "--n-procs", type=int, default=40, help="Number of parallel processes"
    )
    parser.add_argument(
        "--n-feedback-loops", type=int, default=10, help="Number feedback optimization iterations"
    )

    # Boolean Flags
    parser.add_argument(
        "--measure-speedup",
        action="store_true",
        help="Enable runtime speedup measurement",
    )
    parser.add_argument(
        "--validate-final-run",
        action="store_true",
        help="Run exact diagonalization validation on final outputs",
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    full_validation_run(args)

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from autocorr import calc_autocorr
from dotenv import load_dotenv
from pmrqmc_mpi_driver import (
    execute_pt_run_native,
    execute_run_native,
    setup_run_dir,
)
from pt_optimize import optimize_schedule_interpolated
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
    k_max = main_params["k_max"]
    n_feedback_loops = main_params["n_feedback_loops"]
    autocorr_steps = main_params["autocorr_steps"]
    t_exchanges_per_iteration = main_params["t_exchanges_per_iteration"]
    exchanges_per_iteration = main_params["exchanges_per_iteration"]
    measure_speedup = main_params["measure_speedup"]
    validate_final_run = main_params["validate_final_run"]
    test_operator_random_seed = main_params["test_operator_random_seed"]
    schedule_type = main_params["schedule_type"]

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

    autocorr_params = {
        "Tsteps": autocorr_steps,
        "steps": autocorr_steps,
        "stepsPerMeasurement": 1,
        "beta": beta_min,
        "gamma": gamma_min,
        "tau": beta_min / 2,
        "kmax": k_max,
        "Nbins": autocorr_steps,
        "validate": False,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "save_timeseries": True,
        "nproc": 1,
        "test_operator_random_seed": test_operator_random_seed,
    }

    print("executing short run for autocorrelation statistics")
    execute_run_native(run_dir, autocorr_params)

    tau, _bin_length, ok = calc_autocorr(run_dir / "timeseries_rank0.txt")
    print(f"tau was {tau}")
    if not ok:
        print("warning, timeseries is too short")

    print("starting PT temperature optimization")

    init_beta_schedule = np.linspace(beta_max, beta_min, n_temperatures)
    init_gamma_schedule = np.linspace(gamma_max, gamma_min, n_temperatures)

    if schedule_type == "linear_geometric":
        init_gamma_schedule = np.geomspace(gamma_max, gamma_min, n_temperatures)
    elif schedule_type == "geometric_linear":
        init_beta_schedule = np.geomspace(beta_max, beta_min, n_temperatures)

    tau_corr = int(np.ceil(tau))
    instance_details["tau_corr"] = tau_corr

    init_full_schedule = np.array([init_beta_schedule, init_gamma_schedule])
    schedules = [init_full_schedule]

    therm_exchanges = int(t_exchanges_per_iteration )
    tuning_exchanges = int(exchanges_per_iteration )

    params = {
        "Tsteps": therm_exchanges * tau_corr,
        "steps": tuning_exchanges * tau_corr,
        "stepsPerMeasurement": tau_corr,
        "beta": beta_min,
        "beta_range": init_beta_schedule,
        "gamma_range": init_gamma_schedule,
        "gamma": gamma_min,
        "tau": beta_min / 2,
        "kmax": k_max,
        "Nbins": 100,
        "validate": False,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "updates_per_exchange": tau_corr,
        "save_timeseries": False,
        "nproc": n_procs,
        "test_operator_random_seed": test_operator_random_seed,
    }

    avg_rt_length = None

    for iteration in range(n_feedback_loops):
        print(
            f"\n--- Running PT Optimization Iteration {iteration + 1} of {n_feedback_loops} ---"
        )
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

        _, old_schedule, new_schedule = optimize_schedule_interpolated(
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

        schedules.append(new_schedule)

        if np.allclose(old_schedule, new_schedule, rtol=0.01):
            print("Temperature schedule converged. Stopping optimization early.")
            break

    # Target bin length calculation for final production run
    int_rt_length = int(np.ceil(avg_rt_length))
    target_bin_length = int_rt_length * tau_corr * 100
    print(f"Final target bin length: {target_bin_length}")

    instance_details["final_rt_length"] = int_rt_length
    instance_details["target_bin_length"] = target_bin_length

    if measure_speedup:
        autocorr_params = {
            "Tsteps": autocorr_steps,
            "steps": autocorr_steps,
            "stepsPerMeasurement": 1,
            "beta": beta_max,
            "gamma": gamma_max,
            "tau": beta_max / 2,
            "kmax": k_max,
            "Nbins": autocorr_steps,
            "validate": False,
            "save_completed_calculation": False,
            "save_unfinished_calculation": False,
            "resume_calculation": False,
            "save_timeseries": True,
            "nproc": 1,
            "test_operator_random_seed": test_operator_random_seed,
        }
        execute_run_native(run_dir, autocorr_params)
        tau_max_beta, _, _ok = calc_autocorr(run_dir / "timeseries_rank0.txt")
        print(f"tau at max beta was {tau_max_beta}")
        print(
            f"the respective timescales are {tau_max_beta} (single temperature) and {int_rt_length * tau_corr} (PT)"
        )
        print(
            f"the relative speedup is {tau_max_beta / (int_rt_length * tau_corr) / n_temperatures}"
        )

        instance_details["pt_speedup"] = tau_max_beta / (int_rt_length * tau_corr) / n_temperatures

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
        "kmax": k_max,
        "Nbins": final_target_bins,
        "save_completed_calculation": False,
        "save_unfinished_calculation": False,
        "resume_calculation": False,
        "save_timeseries": False,
        "nproc": n_procs,
        "validate": validate_final_run,
        "test_operator_random_seed": test_operator_random_seed,
    }

    print(f"starting final run. expected time: {expected_time} seconds")

    with open(run_dir / "instance_details.json", "w") as f:
        json.dump(instance_details, f)

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
    parser.add_argument("--k-max", type=int, required=True, help="maximum k value")

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
        "--n-feedback-loops",
        type=int,
        default=10,
        help="Number feedback optimization iterations",
    )
    parser.add_argument(
        "--test-operator-random-seed",
        type=int,
        default=42,
        help="seed to generate the test operator",
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

    parser.add_argument(
        "--autocorr-steps",
        type=int,
        help="timeseries length for autocorrelation calculations",
    )
    parser.add_argument(
        "--t-exchanges-per-iteration",
        type=int,
        help="number of thermalization PT exchanges for each optimization iteration",
    )
    parser.add_argument(
        "--exchanges-per-iteration",
        type=int,
        help="number of PT exchanges for each optimization iteration",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    full_validation_run(args)

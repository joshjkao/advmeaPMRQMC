from pathlib import Path

import numpy as np
from autocorr import calc_autocorr
from hamiltonian import generate_1d_tfi, generate_quantum_sk_hamiltonian, toy_3bit
from pmrqmc_mpi_driver import execute_pt_run_native, execute_run_native, setup_run_dir
from pt_optimize import optimize_schedule
from roundtrips import parse_rt_stats

# h_str = toy_3bit()
h_str = generate_1d_tfi(4, 1.0, 1.0, 0.0, True)
# h_str = generate_quantum_sk_hamiltonian(3, 1.0, 1.5, 42)

run_dir = Path("toy_pt_validation")

beta_min = 0.0625
beta_max = 16.0

n_temperatures = 10

autocorr_params = {
    "Tsteps": 10000,
    "steps": 10000,
    "stepsPerMeasurement": 1,
    "beta": beta_min,
    "tau": beta_min / 2,
    "kmax": 5,
    "Nbins": 10000,
    "validate": False,
    "save_completed_calculation": True,
    "save_unfinished_calculation": False,
    "resume_calculation": False,
    "nproc": 1
}

if not run_dir.exists():
    setup_run_dir(run_dir, h_str)

print("executing short run for autocorrelation statistics")
execute_run_native(run_dir, autocorr_params)

tau, bin_length = calc_autocorr(run_dir / "timeseries_rank0.txt")
print(f"tau was {tau}")

print("starting PT temperature optimization")
feedback_loops = 10
init_schedule = np.geomspace(beta_max, beta_min, n_temperatures)
tau_corr = int(np.ceil(tau))

tuning_exchanges = 10000  

params = {
    "Tsteps": 50000,
    "steps": tuning_exchanges * tau_corr,
    "stepsPerMeasurement": tau_corr,
    "beta": beta_min,
    "beta_range": init_schedule,
    "tau": beta_min / 2,
    "kmax": 5,
    "Nbins": 100,
    "validate": False,
    "save_completed_calculation": False,
    "save_unfinished_calculation": False,
    "resume_calculation": False,
    "updates_per_exchange": tau_corr,
    "nproc": 40
}

avg_rt_length = None

for iteration in range(feedback_loops):
    print(f"\n--- Running PT Optimization Iteration {iteration} ---")
    execute_pt_run_native(run_dir, params)
    
    total_visits, round_trips = parse_rt_stats(run_dir / "pmrqmc_pt_flow.csv")
    
    if round_trips > 0:
        avg_rt_length = total_visits / round_trips
        print(f"Iteration {iteration} avg RT length: {avg_rt_length:.2f} exchange attempts")
    else:
        print(f"Warning: 0 round trips completed in iteration {iteration}. Increase tuning_exchanges.")
        avg_rt_length = tuning_exchanges  # Fallback estimate

    _, old_schedule, new_schedule = optimize_schedule(
        run_dir / "pmrqmc_pt_flow.csv", 
        run_dir / "schedule.txt", 
        run_dir / "schedule.txt"
    )
    
    old_betas = old_schedule[:, 0]
    new_betas = new_schedule[:, 0]
    
    params["beta_range"] = new_betas

    if np.allclose(old_betas, new_betas, rtol=0.01):
        print("Temperature schedule converged. Stopping optimization early.")
        break

# Target bin length calculation for final production run
target_bin_length = int(avg_rt_length * tau_corr * 2)
print(f"Final target bin length: {target_bin_length}")

final_target_bins = 1000
final_params = {
    "Tsteps": 20 * target_bin_length,
    "steps": final_target_bins * target_bin_length,
    "stepsPerMeasurement": tau + 1,
    "updates_per_exchange": tau,
    "beta": beta_min,
    "beta_range": params["beta_range"],
    "tau": beta_min / 2,
    "kmax": 5,
    "Nbins": final_target_bins,
    "save_completed_calculation": False,
    "save_unfinished_calculation": False,
    "resume_calculation": False,
    "nproc": 40,
    "validate": True,
}

execute_pt_run_native(run_dir, final_params)

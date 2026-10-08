import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def optimize_schedule(
    flow_csv_path,
    input_schedule_path,
    output_schedule_path,
    epsilon: float = 1e-4
):
    # 1. Load flow data and aggregate directional visits per temperature slot
    df = pd.read_csv(flow_csv_path)
    grouped = df.groupby("temperature", sort=False)[["up_visits", "down_visits"]].first().reset_index()

    # 2. Calculate fraction of upward-moving walkers f(T)
    total_visits = grouped["up_visits"] + grouped["down_visits"]
    f_raw = np.where(total_visits > 0, grouped["up_visits"] / total_visits, 0.0)
    
    # Enforce strict monotonicity (f(T) must decrease from 1.0 at T_min to 0.0 at T_max)
    f = np.maximum.accumulate(f_raw[::-1])[::-1]

    # 3. Compute interval density weights w_i = sqrt(|delta_f| + epsilon)
    delta_f = np.abs(np.diff(f))
    weights = np.sqrt(delta_f + epsilon)

    # Cumulative probability density function (normalized 0 to 1)
    M = len(f)
    cdf = np.zeros(M)
    cdf[1:] = np.cumsum(weights)
    cdf /= cdf[-1]

    # 4. Re-grid schedule uniformly in density space
    target_quantiles = np.linspace(0.0, 1.0, M)
    grid_indices = np.arange(M)
    new_grid_indices = np.interp(target_quantiles, cdf, grid_indices)

    # 5. Read original schedule (supports multi-column files like 'T gamma')
    old_schedule = np.loadtxt(input_schedule_path)
    new_schedule = np.zeros_like(old_schedule)

    # Interpolate each parameter column along the new schedule coordinates
    for col in range(old_schedule.shape[1]):
        new_schedule[:, col] = np.interp(new_grid_indices, grid_indices, old_schedule[:, col])

    # Save new schedule
    np.savetxt(output_schedule_path, new_schedule, fmt="%.6f", delimiter=" ")

    # Display comparison summary
    print(f"{'Index':<6} | {'Old Temp':<10} | {'New Temp':<10} | {'f(T)':<8}")
    print("-" * 42)
    for i in range(M):
        old_t = old_schedule[i, 0] if old_schedule.ndim > 1 else old_schedule[i]
        new_t = new_schedule[i, 0] if new_schedule.ndim > 1 else new_schedule[i]
        print(f"{i:<6} | {old_t:<10.4f} | {new_t:<10.4f} | {f_raw[i]:<8.4f}")

    return f_raw, old_schedule, new_schedule

if __name__ == "__main__":
    # Example usage:
    f_raw, old_sched, new_sched = optimize_schedule(
        flow_csv_path="pmrqmc_pt_flow.csv",
        input_schedule_path="schedule.txt",
        output_schedule_path="schedule.txt"
    )

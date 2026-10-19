import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def optimize_schedule(
    flow_csv_path, input_schedule_path, output_schedule_path, epsilon: float = 1e-4
):
    # 1. Load flow data and aggregate directional visits per temperature slot
    df = pd.read_csv(flow_csv_path)
    grouped = (
        df.groupby("temperature", sort=False)[["up_visits", "down_visits"]]
        .first()
        .reset_index()
    )

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
        new_schedule[:, col] = np.interp(
            new_grid_indices, grid_indices, old_schedule[:, col]
        )

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


def optimize_schedule_interpolated(
    flow_csv_path,
    input_schedule_path,
    output_schedule_path,
    epsilon: float = 1e-4,
):
    # 1. Load flow data
    df = pd.read_csv(flow_csv_path)

    # Group by 'temperature' (which acts as the slot index 0..M-1)
    grouped = (
        df.groupby("temperature", sort=True)[["up_visits", "down_visits"]]
        .first()
        .reset_index()
    )

    # 2. Calculate fraction of upward-moving walkers f(slot)
    total_visits = grouped["up_visits"] + grouped["down_visits"]
    f_raw = np.where(total_visits > 0, grouped["up_visits"] / total_visits, 0.0)

    # Enforce strict monotonicity (f decreases from 1.0 at slot 0 to 0.0 at slot M-1)
    f = np.maximum.accumulate(f_raw[::-1])[::-1]

    # 3. Compute interval density weights w_i = sqrt(|delta_f| + epsilon)
    delta_f = np.abs(np.diff(f))
    weights = np.sqrt(delta_f + epsilon)

    # Cumulative probability density function (normalized 0 to 1)
    M = len(f)
    cdf = np.zeros(M)
    cdf[1:] = np.cumsum(weights)
    cdf /= cdf[-1]

    # 4. Compute new path positions lambda in [0, 1]
    target_quantiles = np.linspace(0.0, 1.0, M)
    old_lambda = np.linspace(0.0, 1.0, M)
    new_lambda = np.interp(target_quantiles, cdf, old_lambda)

    # 5. Load original 2D schedule (M lines, each with 'beta gamma')
    old_schedule = np.atleast_2d(np.loadtxt(input_schedule_path))

    # Verify schedule length matches flow file slot count
    if len(old_schedule) != M:
        raise ValueError(
            f"Schedule length ({len(old_schedule)}) does not match flow file slots ({M})."
        )

    # 6. Linear vector interpolation along the 1D path in (beta, gamma) space
    start_point = old_schedule[0, :]
    end_point = old_schedule[-1, :]

    # Project new_lambda coordinates onto the line connecting start and end points
    new_schedule = start_point + np.outer(new_lambda, (end_point - start_point))

    # Save optimized schedule
    np.savetxt(output_schedule_path, new_schedule, fmt="%.6f", delimiter=" ")

    # Display summary table
    is_2d = old_schedule.shape[1] > 1
    if is_2d:
        print(
            f"{'Slot':<6} | {'Old Beta':<9} | {'New Beta':<9} | {'Old Gamma':<9} | {'New Gamma':<9} | {'f(slot)':<8}"
        )
        print("-" * 64)
        for i in range(M):
            print(
                f"{i:<6} | {old_schedule[i, 0]:<9.4f} | {new_schedule[i, 0]:<9.4f} | "
                f"{old_schedule[i, 1]:<9.4f} | {new_schedule[i, 1]:<9.4f} | {f_raw[i]:<8.4f}"
            )
    else:
        print(f"{'Slot':<6} | {'Old Beta':<10} | {'New Beta':<10} | {'f(slot)':<8}")
        print("-" * 44)
        for i in range(M):
            print(
                f"{i:<6} | {old_schedule[i, 0]:<10.4f} | {new_schedule[i, 0]:<10.4f} | {f_raw[i]:<8.4f}"
            )

    return f_raw, old_schedule, new_schedule


if __name__ == "__main__":
    # Example usage:
    f_raw, old_sched, new_sched = optimize_schedule(
        flow_csv_path="pmrqmc_pt_flow.csv",
        input_schedule_path="schedule.txt",
        output_schedule_path="schedule.txt",
    )

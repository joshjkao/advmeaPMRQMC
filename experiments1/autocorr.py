import sys

import emcee
import numpy as np


def calc_autocorr(file):
    data = np.loadtxt(file)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_tau = 0.0
    num_cols = data.shape[1]
    valid_cols = 0
    ok = True

    for col_index in range(num_cols):
        col = data[:, col_index]

        # Skip constant columns (like a sign without sign-problem)
        if np.all(col == col[0]):
            continue

        valid_cols += 1

        try:
            # c=5 is the standard windowing tolerance for MCMC
            tau = emcee.autocorr.integrated_time(col, c=5)[0]
            max_tau = max(max_tau, tau)

        except emcee.autocorr.AutocorrError as e:
            fallback_tau = e.tau[0] if isinstance(e.tau, (list, np.ndarray)) else e.tau
            max_tau = max(max_tau, fallback_tau)

            print(
                f"Warning (Col {col_index}): Timeseries too short for strict convergence."
            )
            print(f"Using best estimate: tau = {fallback_tau:.2f}")
            ok = False

    if valid_cols == 0:
        print("Warning: No valid fluctuating timeseries were found.")

    if max_tau == 0.0:
        print("Warning: Could not compute autocorrelation for any valid timeseries.")

    recommended_bin_length = int(np.ceil(2 * max_tau))
    return max_tau, recommended_bin_length, ok


if __name__ == "__main__":
    file = sys.argv[1]
    tau, bin_length = calc_autocorr(file)

    print(f"Integrated autocorrelation time (tau): {tau:.2f} steps")
    print(f"Recommended minimum bin_length: {bin_length}")


# 3. Visualize the Autocorrelation Function (ACF) to verify manually
# Calculate ACF up to a lag of 10*tau (or 1000 if tau failed)
# max_lag = int(10 * tau) if 'tau' in locals() else 1000
# acf_values = sm.tsa.acf(data, nlags=max_lag, fft=True)

# plt.figure(figsize=(8, 4))
# plt.plot(acf_values, label="ACF", color="blue")
# plt.axhline(0, color='black', linestyle='--', linewidth=1)

# if 'tau' in locals():
# Plot a vertical line where tau occurs
# plt.axvline(tau, color='red', linestyle='--', label=f'$\\tau = {tau:.1f}$')

# plt.xlabel("Lag (QMC steps)")
# plt.ylabel("Autocorrelation")
# plt.title("Autocorrelation Decay")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()

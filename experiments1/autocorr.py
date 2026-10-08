import sys

import emcee
import numpy as np


def calc_autocorr(file):
    data = np.loadtxt(file)
    data = data[:,-1]
    try:
        # The c=5 parameter is the standard windowing tolerance for MCMC
        tau = emcee.autocorr.integrated_time(data, c=5)[0]
    
        recommended_bin_length=int(np.ceil(2 * tau))

        return tau, recommended_bin_length
    
    except emcee.autocorr.AutocorrError as e:
        print("Warning: The timeseries is too short to reliably measure tau!")
        print(e)

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

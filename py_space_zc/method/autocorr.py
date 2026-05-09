import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal


def autocorr(inp, plot=False, demean=False):
    """
    Calculates the Autocorrelation Function (ACF) and Lag Time for xarray.DataArray.

    Parameters:
    -----------
    inp : xarray.DataArray
        Input scalar data with a 'time' coordinate (np.datetime64).
    plot : bool, optional
        If True, generates a publication-quality plot. Default is False.
    demean : bool, optional
        If True, subtract the mean before correlation. Default is False,
        matching the IDL autocorr_norm.pro convention. Pass an already
        detrended or fluctuating signal when the mean field should be removed.

    Returns:
    --------
    lag_time : numpy.ndarray
        Time lags in seconds.
    acf : numpy.ndarray
        Normalized autocorrelation coefficients (1.0 at lag 0).
    """

    # 1. Data Cleaning
    # Drop NaNs to ensure time and data remain aligned
    inp_clean = inp.dropna(dim="time")

    data = inp_clean.values
    time = inp_clean.time.values

    if len(data) == 0:
        raise ValueError("Input DataArray is empty after dropping NaNs.")

    # 2. Time Conversion: np.datetime64 -> relative seconds
    # Calculation: (T - T0) converted to float seconds
    t_sec = (time - time[0]) / np.timedelta64(1, 's')
    dt = np.median(np.diff(t_sec))

    # 3. Optional mean subtraction
    # Default follows IDL autocorr_norm.pro: no additional demeaning here.
    data_corr = data - np.mean(data) if demean else data

    # 4. Compute Autocorrelation
    # signal.correlate is computationally efficient for large arrays.
    # IDL formula:
    # A_L = N/(N-L) * sum(X[0:N-L] * X[L:N]) / sum(X^2)
    n = len(data_corr)
    corr = signal.correlate(data_corr, data_corr, mode='full', method='fft')

    norm_factor = np.sum(data_corr ** 2)
    if not np.isfinite(norm_factor) or norm_factor <= 0:
        raise ValueError("Input data have zero or invalid power.")

    # Extract positive lags only
    lags = np.arange(0, n)
    acf = corr[n - 1:] / norm_factor
    acf *= n / (n - lags)
    acf[0] = 1.0
    lag_time = lags * dt

    # 5. Conditional Plotting
    if plot:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot ACF using Tab10 Blue
        ax.plot(lag_time, acf, color='tab:blue', linewidth=2, label='Autocorrelation')

        # Reference Lines
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(np.exp(-1), color='gray', linestyle='--', linewidth=1, label='$e^{-1}$ Threshold')

        # Calculate Coherence Time (where ACF drops below 1/e)
        # 1/e is approx 0.368
        below_e = np.where(acf <= np.exp(-1))[0]
        if len(below_e) > 0:
            idx_e = below_e[0]
            t_coh = lag_time[idx_e]
            ax.axvline(t_coh, color='tab:red', linestyle=':', linewidth=1.5)
            ax.text(t_coh * 1.1, 0.8, f'Coherence Time: {t_coh:.1f} s',
                    color='tab:red', fontweight='bold', family='serif')

        # Aesthetics
        ax.set_xlabel('Lag Time (s)', fontsize=14, family='serif')
        ax.set_ylabel('ACF', fontsize=14, family='serif')
        ax.set_title('Autocorrelation Analysis', fontsize=16, family='serif', fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        # Focus on the relevant part (first 25% of the total duration)
        ax.set_xlim([0, lag_time[-1] / 4])
        ax.set_ylim([-0.2, 1.1])

        ax.legend(frameon=False, loc='upper right')
        ax.tick_params(direction='in', which='both', top=True, right=True)

        plt.tight_layout()
        plt.show()

    return lag_time, acf

# --- Usage Example ---
# lag, acf_values = calculate_autocorr(b_magnitude_xr, plot=True)

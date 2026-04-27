import numpy as np
from scipy.interpolate import interp1d

def ei_rate_Te(Te_K, species):
    """
    Get electron impact ionization rate coefficient for a given species and electron temperature.
    From
    Cravens, T. E., Kozyra, J. U., Nagy, A. F., Gombosi, T. I., & Kurtz, M. (1987).
    Electron impact ionization in the vicinity of comets.
    Journal of Geophysical Research: Space Physics, 92(A7), 7341-7353.
    https://doi.org/10.1029/JA092iA07p07341
    Parameters:
    -----------
    Te_K : float or np.ndarray
        Electron temperature(s) in Kelvin. from 1e4 - 1e7 K, about 1eV - 1000 eV
    species : str
        Ion species. One of 'H', 'O', 'CO', 'CO2'.

    Returns:
    --------
    rate_m3s : float or np.ndarray
        Ionization rate coefficient [m^3/s].
    """
    # Electron temperature grid [K]
    Temp_e = np.array((1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,
                      1.5e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6,
                      1.5e6, 2e6, 3e6, 4e6, 5e6, 7e6, 1e7))

    # Ionization rates [cm^3/s]
    rates_cm3s = {
        'CO2': np.array([1.69e-17, 7.617e-13, 3.067e-11, 2.054e-10, 6.625e-10, 1.477e-9,
                         2.660e-9, 4.185e-9, 6.006e-9, 8.08e-9, 2.083e-8, 3.515e-8, 6.233e-8,
                         8.64e-8, 1.063e-7, 1.229e-7, 1.369e-7, 1.488e-7, 1.59e-7, 1.679e-7,
                         1.979e-7, 2.145e-7, 2.301e-7, 2.352e-7, 2.362e-7, 2.328e-7, 2.248e-7]),
        'CO': np.array([2.325e-16, 1.666e-12, 4.125e-11, 2.285e-10, 6.758e-10, 1.441e-9, 2.532e-9,
                        3.926e-9, 5.587e-9, 7.471e-9, 1.904e-8, 3.197e-8, 5.664e-8, 7.761e-8,
                        9.487e-8, 1.091e-7, 1.208e-7, 1.306e-7, 1.389e-7, 1.458e-7,
                        1.683e-7, 1.793e-7, 1.867e-7, 1.863e-7, 1.83e-7, 1.749e-7, 1.63e-7]),
        'O': np.array([4.682e-16, 2.172e-12, 4.547e-11, 2.288e-10, 6.318e-10, 1.275e-9,
                      2.14e-9, 3.188e-9, 4.36e-9, 5.684e-9, 1.29e-8, 2.008e-8, 3.241e-8,
                      4.201e-8, 4.955e-8, 5.554e-8, 6.047e-8, 6.452e-8, 6.791e-8, 7.078e-8,
                      8.006e-8, 8.466e-8, 8.789e-8, 8.774e-8, 8.625e-8, 8.257e-8, 7.709e-8]),
        'H': np.array([1.9e-15, 4.24e-12, 6.19e-11, 2.49e-10, 5.93e-10, 1.08e-9, 1.67e-9,
                      2.34e-9, 3.06e-9, 3.82e-9, 7.64e-9, 1.11e-8, 1.65e-8, 2.05e-8,
                      2.33e-8, 2.54e-8, 2.69e-8, 2.81e-8, 2.89e-8, 2.95e-8, 3.03e-8,
                      2.95e-8, 2.67e-8, 2.41e-8, 2.2e-8, 1.88e-8, 1.56e-8])
    }

    species = species.upper()
    if species not in rates_cm3s:
        raise ValueError(f"Invalid species '{species}'. Must be one of: 'H', 'O', 'CO', 'CO2'.")

    # Interpolation function
    interp_func = interp1d(Temp_e, rates_cm3s[species], kind='linear',
                           bounds_error=False, fill_value='extrapolate')

    rate_cm3s = interp_func(Te_K)
    rate_m3s = rate_cm3s * 1e-6  # Convert cm^3/s → m^3/s

    return rate_m3s


if __name__ == '__main__':
    T = np.linspace(1e4, 1e7, num = 1000)
    rate_H = ei_rate_Te(T, species='H')
    rate_O = ei_rate_Te(T, species='O')
    rate_CO2 = ei_rate_Te(T, species='CO2')
    rate_CO = ei_rate_Te(T, species='CO')
    import matplotlib.pyplot as plt
    from py_space_zc import plot

    fig, axs = plot.subplot(1, 1, figsize=(8, 6))

    # Plot log-log curves with LaTeX labels
    axs[0].loglog(T, rate_CO2, color='blue', label=r'$\mathrm{CO}_2^+$')
    axs[0].loglog(T, rate_CO, color='purple', label=r'$\mathrm{CO}^+$')
    axs[0].loglog(T, rate_O, color='red', label=r'$\mathrm{O}^+$')
    axs[0].loglog(T, rate_H, color='green', label=r'$\mathrm{H}^+$')

    # Labels and appearance
    axs[0].set_xlabel(r'Electron Temperature [K]', fontsize=13)
    axs[0].set_ylabel(r'Ionization Rate [m$^3$/s]', fontsize=13)
    axs[0].set_title(r'Electron Impact Ionization Rates', fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True, which='both', linestyle='--', alpha=0.5)
    axs[0].set_xlim(1e4, 1e7)
    axs[0].set_ylim(1e-16, 1e-12)

    # Styling
    plot.set_axis(axs[0], fontsize=12, tick_fontsize=12, label_fontsize=13)
    plt.show()
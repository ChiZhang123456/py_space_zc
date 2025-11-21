import numpy as np
import matplotlib.pyplot as plt


def mars_neutral_density(alt_m, case='issi_solar_max'):
    """
    Compute neutral densities (H, O, CO2) on Mars for a given solar condition
    using multi-exponential empirical models.

    Parameters
    ----------
    alt_m : float or ndarray
        Altitude in meters.
    case : str
        Solar activity case:
        'solarmin', 'solarmod', 'solarmax', or 'issi_solar_max', 'issi_solar_min'
        'solarmin', 'solarmod', 'solarmax' cases are somewhat unreliable

    Returns
    -------
    nH, nO, nCO2 : ndarray
        Neutral hydrogen, oxygen, and CO2 densities [m^-3].
    """

    # Convert altitude from meters → kilometers
    alt_km = np.asarray(alt_m) / 1000.0

    # Standard multi-exponential neutral profiles (cm^-3, scale heights in km)
    profiles = {
        'solarmin': {
            'n0': {'CO2': 1.1593e12, 'CO2x': 2.2258e11,
                   'O': 3.2278e9, 'Ox': 5.2695e8, 'Oh': 1.951e4, 'Ohx': 1.5248e3,
                   'H': 1.1307e7, 'Hx': 9.4936e5},
            'H': {'CO2': 5.2667, 'CO2x': 10.533,
                  'O': 9.486, 'Ox': 30.45, 'Oh': 290.5, 'Ohx': 2436.6,
                  'H': 13.133, 'Hx': 586.6}
        },

        'solarmod': {
            'n0': {'CO2': 4.435e12, 'CO2x': 8.0807e10,
                   'O': 8.0283e9, 'Ox': 5.1736e8, 'Oh': 6.3119e4, 'Ohx': 3.9646e3,
                   'H': 1.8374e6, 'Hx': 7.3638e4},
            'H': {'CO2': 6.5631, 'CO2x': 17.064,
                  'O': 13.34, 'Ox': 50.025, 'Oh': 290.5, 'Ohx': 2436.6,
                  'H': 13.133, 'Hx': 610.0}
        },

        'solarmax': {
            'n0': {'CO2': 4.435e12, 'CO2x': 8.0807e10,
                   'O': 8.0283e9, 'Ox': 5.1736e8, 'Oh': 6.3119e4, 'Ohx': 3.9646e3,
                   'H': 1.8374e6, 'Hx': 7.3638e4},
            'H': {'CO2': 6.5631, 'CO2x': 17.064,
                  'O': 13.34, 'Ox': 50.025, 'Oh': 290.5, 'Ohx': 2436.6,
                  'H': 13.133, 'Hx': 610.0}
        },
    }

    # -------------------------------
    # Special ISSI solar-maximum model
    # -------------------------------
    if case == "issi_solar_max":
        nCO2 = (5.88e18 * np.exp(-alt_km / 7.00) +
                3.55e13 * np.exp(-alt_km / 16.67)) * 1e6

        nO_cold = (2.33e13 * np.exp(-alt_km / 12.27) +
                   2.84e09 * np.exp(-alt_km / 48.57)) * 1e6

        nO_hot = (1.56e4 * np.exp(-alt_km / 696.9) +
                  2.92e3 * np.exp(-alt_km / 2891.) +
                  5.01e4 * np.exp(-alt_km / 99.19)) * 1e6

        nH_cold = (1e3 * np.exp(9.25e5 * (1/(alt_km+3393.5) - 1/3593.5))) * 1e6
        nH_hot  = (3e4 * np.exp(1.48e4 * (1/(alt_km+3393.5) - 1/3593.5))) * 1e6
        return (nH_cold + nH_hot, nO_cold + nO_hot, nCO2)

    elif case == "issi_solar_min":
        nCO2 = (6.04e18 * np.exp(-alt_km / 6.98) +
                1.67e15 * np.exp(-alt_km / 11.49)) * 1e6
        nO = (5.85e13 * np.exp(-alt_km / 10.56) +
              7.02e9 * np.exp(-alt_km / 33.97) +
              5.23e3 * np.exp(-alt_km/626.2) +
              9.76e2 * np.exp(-alt_km/2790) +
              3.71e4 * np.exp( -alt_km/ 88.47)) * 1e6
        nH = (1.5e5 * np.exp(25965 * (1/(alt_km+3393.5) - 1/3593.5)) +
              1.9e4 * np.exp(10365 * (1/(alt_km+3393.5) - 1/3593.5))) * 1e6
        return nH, nO, nCO2

    # If not ISSI model, must be in profiles dict
    if case not in profiles:
        raise ValueError("Invalid case. Choose 'solarmin', 'solarmod', 'solarmax', or 'issi_solar_max', 'issi_solar_min'")

    # Retrieve empirical parameters
    p = profiles[case]
    n0, H_km = p['n0'], p['H']

    # Convert cm^-3 → m^-3 (multiply by 1e6)
    nCO2 = (n0['CO2']  * np.exp(-alt_km / H_km['CO2']) +
            n0['CO2x'] * np.exp(-alt_km / H_km['CO2x'])) * 1e6

    nO = (n0['O']   * np.exp(-alt_km / H_km['O']) +
          n0['Ox']  * np.exp(-alt_km / H_km['Ox']) +
          n0['Oh']  * np.exp(-alt_km / H_km['Oh']) +
          n0['Ohx'] * np.exp(-alt_km / H_km['Ohx'])) * 1e6

    nH = (n0['H']  * np.exp(-alt_km / H_km['H']) +
          n0['Hx'] * np.exp(-alt_km / H_km['Hx'])) * 1e6

    return nH, nO, nCO2


# =====================================================================
#                      PLOT COMPARISON: ISSI vs SOLARMAX
# =====================================================================
if __name__ == '__main__':
    alt = np.linspace(0, 1000e3, 2000)

    nH_smax, nO_smax, nCO2_smax = mars_neutral_density(alt, "solarmax")
    nH_issi, nO_issi, nCO2_issi = mars_neutral_density(alt, "issi_solar_max")

    fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)

    species = ["CO2", "O", "H"]
    solarmax_profiles = [nCO2_smax, nO_smax, nH_smax]
    issi_profiles     = [nCO2_issi, nO_issi, nH_issi]

    for i in range(3):
        ax[i].semilogx(solarmax_profiles[i], alt/1000, label="solarmax", lw=2)
        ax[i].semilogx(issi_profiles[i], alt/1000, label="ISSI solar max", lw=2, ls="--")

        ax[i].set_xlabel(f"{species[i]} density [m$^{{-3}}$]")
        ax[i].grid(True, which="both", ls="--", alpha=0.4)
        ax[i].set_xlim(1e0, 1e14)

    ax[0].set_ylabel("Altitude [km]")
    fig.suptitle("Comparison: Standard Solar-Max vs ISSI Solar-Max Neutral Profiles")
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

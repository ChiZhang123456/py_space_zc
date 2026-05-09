import numpy as np
import matplotlib.pyplot as plt
import os


def mars_neutral_density(alt_m, case='issi_solar_max'):
    """
    Compute neutral densities in the Martian atmosphere or exosphere.

    Parameters
    ----------
    alt_m : float or ndarray
        Altitude in meters.
    case : str
        Neutral density case.
        Use 'ra', 'rahmati', or 'rahmati_exosphere' for the H and O exosphere
        profile stored in Rahmati_exosphere.txt. This reproduces the data
        lookup in mars_exosphere.m. Only H and O are available for this case.
        Use 'issi_solar_max' or 'issi_solar_min' for the ISSI model.
        Use 'solarmin', 'solarmod', or 'solarmax' for the older empirical
        models. These cases are somewhat unreliable.

    Returns
    -------
    nH, nO : ndarray
        Returned for case='ra'. Neutral hydrogen and oxygen exosphere
        densities [m^-3].
    nH, nO, nCO2 : ndarray
        Returned for other cases. Neutral hydrogen, oxygen, and CO2 densities
        [m^-3].

    Examples
    --------
    >>> nH, nO = mars_neutral_density(4000e3, case='ra')
    >>> nH, nO, nCO2 = mars_neutral_density(400e3, case='issi_solar_max')
    """

    # Convert altitude from meters → kilometers
    case = case.lower()

    alt_km = np.asarray(alt_m) / 1000.0

    if case in ("ra", "rahmati", "rahmati_exosphere"):
        return _rahmati_exosphere_density(alt_km)

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
        raise ValueError("Invalid case. Choose 'ra', 'solarmin', 'solarmod', 'solarmax', 'issi_solar_max', or 'issi_solar_min'")

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


def _rahmati_exosphere_density(alt_km):
    """
    Return H and O exosphere densities from Rahmati_exosphere.txt.

    The table stores altitude in km and density in cm^-3. The returned
    densities are converted to m^-3 to keep the unit convention of
    mars_neutral_density. Linear interpolation and extrapolation are used to
    match MATLAB interp1(..., 'linear', 'extrap').
    """
    filename = os.path.join(os.path.dirname(__file__), "Rahmati_exosphere.txt")
    data = np.loadtxt(filename, ndmin=2)

    h_good = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    o_good = np.isfinite(data[:, 2]) & np.isfinite(data[:, 3])

    alt_H = data[h_good, 0]
    nH_cm3 = data[h_good, 1]
    alt_O = data[o_good, 2]
    nO_cm3 = data[o_good, 3]

    nH = _interp_linear_extrap(alt_km, alt_H, nH_cm3) * 1e6
    nO = _interp_linear_extrap(alt_km, alt_O, nO_cm3) * 1e6

    return nH, nO


def _interp_linear_extrap(x, xp, fp):
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_work = np.atleast_1d(x_arr)

    xp = np.asarray(xp, dtype=float).reshape(-1)
    fp = np.asarray(fp, dtype=float).reshape(-1)
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]

    if xp.size < 2:
        raise ValueError("The exosphere profile must contain at least two points.")

    out = np.interp(x_work, xp, fp)
    low = x_work < xp[0]
    high = x_work > xp[-1]
    out[low] = fp[0] + (x_work[low] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    out[high] = fp[-1] + (x_work[high] - xp[-1]) * (fp[-1] - fp[-2]) / (
        xp[-1] - xp[-2]
    )

    if scalar_input:
        return float(out[0])
    return out


# =====================================================================
#                      PLOT COMPARISON: ISSI vs SOLARMAX
# =====================================================================
if __name__ == '__main__':
    # 将高度扩展到 5000 km，以观察外逸层 H 和 O 的分布
    alt = np.linspace(100e3, 5000e3, 1000)
    alt_km = alt / 1000.0

    # 获取三种模型的数据
    nH_smax, nO_smax, nCO2_smax = mars_neutral_density(alt, "solarmax")
    nH_issi, nO_issi, nCO2_issi = mars_neutral_density(alt, "issi_solar_max")

    # 注意：Rahmati 模型只返回 nH, nO。这里假设文件存在，如果不存在请跳过此步
    try:
        nH_ra, nO_ra = mars_neutral_density(alt, "ra")
        has_ra = True
    except:
        print("Warning: Rahmati_exosphere.txt not found. Skipping RA model.")
        has_ra = False

    fig, ax = plt.subplots(1, 3, figsize=(16, 7), sharey=True)

    # 1. CO2 比较 (通常 Rahmati 没有 CO2)
    ax[0].semilogx(nCO2_smax, alt_km, label="Standard SolarMax", lw=2)
    ax[0].semilogx(nCO2_issi, alt_km, label="ISSI SolarMax", lw=2, ls="--")
    ax[0].set_xlabel("$CO_2$ Density [m$^{-3}$]")
    ax[0].set_xlim(1e6, 1e20)  # CO2 在低层大气浓度极高

    # 2. Oxygen (O) 比较
    ax[1].semilogx(nO_smax, alt_km, label="Standard SolarMax", lw=2)
    ax[1].semilogx(nO_issi, alt_km, label="ISSI SolarMax", lw=2, ls="--")
    if has_ra:
        ax[1].semilogx(nO_ra, alt_km, label="Rahmati (RA)", lw=2, ls="-.")
    ax[1].set_xlabel("O Density [m$^{-3}$]")
    ax[1].set_xlim(1e4, 1e16)

    # 3. Hydrogen (H) 比较
    ax[2].semilogx(nH_smax, alt_km, label="Standard SolarMax", lw=2)
    ax[2].semilogx(nH_issi, alt_km, label="ISSI SolarMax", lw=2, ls="--")
    if has_ra:
        ax[2].semilogx(nH_ra, alt_km, label="Rahmati (RA)", lw=2, ls="-.")
    ax[2].set_xlabel("H Density [m$^{-3}$]")
    ax[2].set_xlim(1e4, 1e12)

    # 格式美化
    ax[0].set_ylabel("Altitude [km]")
    for a in ax:
        a.grid(True, which="both", ls="--", alpha=0.5)
        a.set_ylim(100, 5000)

    fig.suptitle("Comparison of Mars Neutral Models (Solar Max Conditions)", fontsize=16)
    ax[2].legend(loc="upper right", frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
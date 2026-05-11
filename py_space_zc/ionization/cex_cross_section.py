import numpy as np
import matplotlib.pyplot as plt
import os


def _read_x1e16_table(filename):
    rows = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            stripped = ln.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split(',') if ',' in stripped else stripped.split()
            try:
                rows.append([float(part) for part in parts[:2]])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric data lines found in: {filename}")

    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1] * 1e-16


def _table_cex_cross_section(energy_eV, table_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, table_name)
    energy_table, sigma_table = _read_x1e16_table(filename)

    E = np.asarray(energy_eV, dtype=float)
    sigma = np.interp(E, energy_table, sigma_table, left=np.nan, right=np.nan)
    if np.isscalar(energy_eV):
        return float(np.asarray(sigma))
    return sigma

def cex_cross_section(energy_eV, ion = 'H+', neutral = 'H'):
    """
    Charge-exchange collision cross section (cm^2).

    Parameters
    ----------
    energy_eV : float or array-like
        Ion kinetic energy in eV.
    ion : str
        Ion species, e.g. 'H+', 'O+'.
    neutral : str
        Neutral species, e.g. 'H', 'O', 'N2', 'O2'.

    Returns
    -------
    sigma_cm2 : ndarray or float
        Cross section in cm^2 (same shape as energy_eV).

    Notes
    -----
    Reactions and sources:
      - H+ + O   -> H + O+   (HpO>OpH)
      - H+ + H   -> H + H+   (HpH>HpH)
      - O+ + H   -> H+ + O   (OpH>HpO)
      - O+ + O   -> O+ + O   (OpO>OpO)
      - H+ + N2  -> N2+ + H  (HpN2>N2PH)
      - H+ + O2  -> O2+ + H  (HpO2>O2PH)
      - H+ + Na/Mg/K/Ca -> metal+ + H, digitized from Lin and Poppe
        (2025), Fig. 2c.
      - H+ + Al/Si/Ti uses the "Others" average curve from Lin and Poppe
        (2025), Fig. 2c, following their treatment for metals without
        species-specific charge-exchange data.

    Formulae follow the MATLAB reference you provided:
    Lindsay & Stebbings (2005), https://doi.org/10.1029/2005JA011298
    """
    # normalize species strings
    ion = ion.strip().lower().replace(' ', '')
    neutral = neutral.strip().lower().replace(' ', '')

    if ion == 'h+':
        metal_table_map = {
            'na': 'charge_exchange_Hp_Na.txt',
            'mg': 'charge_exchange_Hp_Mg.txt',
            'k': 'charge_exchange_Hp_K.txt',
            'ca': 'charge_exchange_Hp_Ca.txt',
            'al': 'charge_exchange_Hp_Others.txt',
            'si': 'charge_exchange_Hp_Others.txt',
            'ti': 'charge_exchange_Hp_Others.txt',
            'others': 'charge_exchange_Hp_Others.txt',
            'other': 'charge_exchange_Hp_Others.txt',
        }
        table_name = metal_table_map.get(neutral)
        if table_name is not None:
            return _table_cex_cross_section(energy_eV, table_name)

    # map to a reaction key
    if ion == 'h+' and neutral == 'o':
        key = 'HpO>OpH'
    elif ion == 'h+' and neutral == 'h':
        key = 'HpH>HpH'
    elif ion == 'o+' and neutral == 'h':
        key = 'OpH>HpO'
    elif ion == 'o+' and neutral == 'o':
        key = 'OpO>OpO'
    elif ion == 'h+' and neutral == 'n2':
        key = 'HpN2>N2PH'
    elif ion == 'h+' and neutral == 'o2':
        key = 'HpO2>O2PH'
    else:
        raise ValueError(f"Unsupported reaction for ion='{ion}', neutral='{neutral}'.")

    E = np.asarray(energy_eV, dtype=float)  # eV
    # avoid invalid logs/divisions by using safe helpers
    # but keep the same algebra as in the MATLAB code
    logEk = np.log(E / 1e3)  # log(E/1e3)

    if key in ('HpO>OpH',):
        a1, a2, a3 = 2.91, 0.0886, 50.9
        a4, a5, a6 = 4.73, -0.862, 0.0306
        term1 = (a1 - a2 * logEk)**2 * (1 - np.exp(-a3 / (E / 1e3)))**2
        term2 = (a4 - a5 * logEk) * (1 - np.exp(-a6 / (E / 1e3)))**2
        res = term1 + term2

    elif key in ('HpH>HpH',):
        a1, a2, a3 = 4.15, 0.531, 67.3
        res = (a1 - a2 * logEk)**2 * (1 - np.exp(-a3 / (E / 1e3)))**4.5

    elif key in ('OpH>HpO',):
        a1, a2, a3 = 3.13, 0.17, 87.5
        res = (a1 - a2 * logEk)**2 * (1 - np.exp(-a3 / (E / 1e3)))**0.8

    elif key in ('OpO>OpO',):
        a1, a2, a3 = 4.07, 0.269, 415.0
        res = (a1 - a2 * logEk)**2 * (1 - np.exp(-a3 / (E / 1e3)))**0.8

    elif key in ('HpN2>N2PH',):
        # H+ + N2 -> N2+ + H
        a1, a2, a3 = 12.5, 1.52, 3.97
        a4, a5, a6, a7 = 0.36, -1.2, 0.208, 0.741
        termA = a1 * np.exp(-(np.log(E/1e3)-a2)**2/ a3) * (1 - np.exp(-(E/1e3) / a4))**2
        termB = (a5 - a6 * logEk)**2 * (1 - np.exp(-a7 * 1e3 / E))**2
        res = termA + termB

    elif key in ('HpO2>O2PH',):
        # H+ + O2 -> O2+ + H
        a1, a2, a3 = 1.83, -0.545, 15.8
        a4, a5, a6 = 6.35, -0.801, 0.24
        # Note: This follows your MATLAB algebra literally:
        term1 = (a1 - a2 * logEk)**2 * (1 - np.exp(-a3 * 1e3 / E))**1.5
        term2 = (a4 - a5 * logEk)**2 * (1 - np.exp(-a6 * 1e3 / E))
        res = term1 + term2

    else:
        raise RuntimeError("Unexpected reaction key.")

    # convert to cm^2 as in the MATLAB code (multiply by 1e-16)
    sigma_cm2 = res * 1e-16

    # return scalar if input was scalar
    if np.isscalar(energy_eV):
        return float(np.asarray(sigma_cm2))
    return sigma_cm2


if __name__ == '__main__':
    energy = np.linspace(1.0, 10000.0, 10000)
    cs_h = cex_cross_section(energy, ion = 'H+', neutral = 'H')
    cs_o = cex_cross_section(energy, ion='H+', neutral='O')
    cs_o_o = cex_cross_section(energy, ion='O+', neutral='O')
    cs_o_h = cex_cross_section(energy, ion='O+', neutral='H')
    cs_o2 = cex_cross_section(energy, ion='H+', neutral='O2')
    cs_n2 = cex_cross_section(energy, ion='H+', neutral='n2')

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    ax.plot(energy, cs_h, label='H$^+$ + H', linewidth=2)
    ax.plot(energy, cs_o, label='H$^+$ + O', linewidth=2)
    ax.plot(energy, cs_o_o, label='O$^+$ + O', linewidth=2)
    ax.plot(energy, cs_o_h, label='O$^+$ + H', linewidth=2)

    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Charge-exchange cross section (cm$^2$)', fontsize=12)


    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Charge-exchange cross section (cm$^2$)', fontsize=12)
    plt.title('Charge-exchange Cross Sections vs Energy', fontsize=13)
    plt.legend(frameon=False)
    plt.grid(True, which='both', alpha=0.3)
    # ax.set_ylim(0*1e-15, 7*1e-15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

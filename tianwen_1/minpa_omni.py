from py_space_zc.tianwen_1.get_data import get_data
import numpy as np
import py_space_zc
from scipy.interpolate import interp1d
from py_space_zc.vdf import rebin_omni

def minpa_omni(tint, species):
    """
    Extracts and computes omni-directional (angle-integrated) DEF
    for a given ion species from 'minpa_mod1' data, interpolated onto
    a regular 16-second cadence and filtered for large time gaps.

    Parameters
    ----------
    tint : list of str
        Time interval, e.g., ['2022-10-18T13:08', '2022-10-18T13:12']

    species : str
        Ion species name: 'H', 'O', 'O2', 'CO2' (with or without '+')

    Returns
    -------
    res : xarray.DataArray
        Omni-directional DEF spectrogram with time and energy coordinates.
    """

    # Mapping from species name to DEF mass index
    species_mass_index = {
        'h': 1, 'h+': 1,
        'o': 4, 'o+': 4,
        'o2': 6, 'o2+': 6,
        'co2': 7, 'co2+': 7,
    }

    # Validate species
    sp = species.lower()
    if sp not in species_mass_index:
        raise ValueError(f"Unsupported species: {species}")

    # Load minpa_mod1 data
    minpa = get_data(tint, 'minpa_mod1')
    mass_index = species_mass_index[sp]

    # Extract DEF and compute omni-directional DEF
    def_4d = np.squeeze(minpa["DEF"][:, :, :, :, mass_index])  # [time, energy, phi, theta]
    def_omni = np.nansum(def_4d, axis=(2, 3))  # [time, energy]

    # Original time and energy
    time_orig = minpa["time"].astype('datetime64[s]')
    energy_orig = minpa["energy"]

    # Convert original time to seconds since epoch
    time_orig_sec = time_orig.astype(float)

    # Define new uniform time array at 16-second resolution
    dt_sec = 16
    t_start = time_orig_sec[0]
    t_end = time_orig_sec[-1]
    time_uniform_sec = np.arange(t_start, t_end + dt_sec, dt_sec)
    time_uniform = time_uniform_sec.astype('datetime64[s]')

    # Interpolate DEF
    interp_func = interp1d(time_orig_sec, def_omni, axis=0,
                           kind='linear', bounds_error=False, fill_value=np.nan)
    def_interp = interp_func(time_uniform_sec)

    # Interpolate energy if needed (optional: assumes slow variation)
    energy_interp_func = interp1d(time_orig_sec, energy_orig,
                                  axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
    energy_interp = energy_interp_func(time_uniform_sec)

    # Remove interpolated points that are too far from original time points (>60s)
    min_diff = np.min(np.abs(time_uniform_sec[:, None] - time_orig_sec[None, :]), axis=1)
    def_interp[min_diff > 60] = np.nan

    # Package into xarray object via py_space_zc
    res = py_space_zc.ts_spectr(
        time=time_uniform,
        ener=energy_interp,
        data=def_interp,
        comp_name='energy',
        attrs={
            "UNITS": "keV/(cm^2 s sr keV)",
            "species": species.upper()
        }
    )
    res = rebin_omni(res, scpot=None)
    return res

# === Example usage ===
if __name__ == "__main__":
    tint = ["2022-10-18T13:08", "2022-10-18T13:12"]
    H_omni = minpa_omni(tint, 'H')
    print(H_omni)

from py_space_zc import maven, vdf, ts_scalar
import numpy as np


def c6_density(tint, correct_bkg=False):
    """
    Compute number densities of H+, O+, and O2+ from MAVEN STATIC C6 mode.

    Parameters
    ----------
    tint : list of str
        Time interval in ISO format,
        e.g., ["2015-01-18T01:40", "2015-01-18T02:25"]

    correct_bkg : bool, optional
        Whether to apply background correction to STATIC C6 DEF data.
        Default is False.

    Returns
    -------
    nH : ts_scalar
        H+ number density [cm^-3]
    nO : ts_scalar
        O+ number density [cm^-3]
    nO2 : ts_scalar
        O2+ number density [cm^-3]
    """

    # ------------------------------------------------------------------
    # Load STATIC C6 data
    # ------------------------------------------------------------------
    c6 = maven.get_data(tint, "static_c6")

    # ------------------------------------------------------------------
    # Extract DEF and convert to PSD
    # ------------------------------------------------------------------
    H_psd = vdf.flux_convert(
        maven.static.extract_data_c6(tint, "H", correct_background=correct_bkg),
        "def2psd"
    )
    O_psd = vdf.flux_convert(
        maven.static.extract_data_c6(tint, "O", correct_background=correct_bkg),
        "def2psd"
    )
    O2_psd = vdf.flux_convert(
        maven.static.extract_data_c6(tint, "O2", correct_background=correct_bkg),
        "def2psd"
    )


    # Get mass and charge
    mass_H, qH   = vdf.get_particle_mass_charge("H")
    mass_O, qO   = vdf.get_particle_mass_charge("O")
    mass_O2, qO2 = vdf.get_particle_mass_charge("O2")

    # Angular integration constants
    theta = 0
    deltaphi = 2 * np.pi

    # dE and energy (corrected for spacecraft potential)
    dE = c6["denergy"] * qH
    energy_corrected = c6["energy"] - c6["scpot"].data[:, None]
    energy_corrected = np.where(energy_corrected > 0, energy_corrected, np.nan)
    energy = energy_corrected * qH

    # Geometric + integration coefficient
    theta_width = np.deg2rad(c6["dtheta"] / 2)
    common_term = 2 ** 1.5 * np.cos(theta) * np.sin(theta_width) * deltaphi

    coff_H  = common_term * energy**0.5 * dE * mass_H**(-1.5)
    coff_O  = common_term * energy**0.5 * dE * mass_O**(-1.5)
    coff_O2 = common_term * energy**0.5 * dE * mass_O2**(-1.5)

    # Compute number density [cm^-3]
    nH  = ts_scalar(H_psd.time.data,  np.nansum(H_psd * coff_H,  axis=1) / 1e6)
    nO  = ts_scalar(O_psd.time.data,  np.nansum(O_psd * coff_O,  axis=1) / 1e6)
    nO2 = ts_scalar(O2_psd.time.data, np.nansum(O2_psd * coff_O2, axis=1) / 1e6)

    return nH, nO, nO2

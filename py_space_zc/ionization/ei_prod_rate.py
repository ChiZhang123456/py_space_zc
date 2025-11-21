import numpy as np
from py_space_zc.ionization import ei_cross_section


def ei_prod_rate(energy, DEF,
                 scpot=None,
                 dtheta=np.pi/2,
                 dphi=2 * np.pi,
                 species='H'):
    """
    Compute electron impact ionization production rate from OMNI differential energy flux (DEF).

    Parameters
    ----------
    energy : np.ndarray of shape (n_energy,)
        Energy bins in eV (must be positive and monotonically increasing).

    DEF : np.ndarray of shape (n_time, n_energy)
        Differential energy flux (typically in eV⁻¹ cm⁻² s⁻¹ sr⁻¹).

    scpot : np.ndarray of shape (n_time,) or None, optional
        Spacecraft potential (in eV). If None, assumed zero. Units: [eV].

    dtheta : float, optional
        Angular width in theta direction (in radians). Default: pi/2 (hemisphere).

    dphi : float, optional
        Angular width in phi direction (in radians). Default: 2*pi (full azimuth).

    species : str
        Target species for electron impact (e.g., 'H', 'O2', 'CO2').

    Returns
    -------
    prod_rate : np.ndarray of shape (n_time,)
        Ionization production rate (in units of cm⁻³ s⁻¹).

    Notes
    -----
    - This function interpolates cross section data using ei_cross_section().
    - Energy below 10 eV is ignored in the calculation.
    """

    n_time, n_energy = DEF.shape

    # Construct 2D energy matrix for each time step
    energy_mat = np.tile(energy[None, :], (n_time, 1))  # shape: (n_time, n_energy)

    # Spacecraft potential correction (if any)
    if scpot is None:
        scpot_mat = np.zeros_like(energy_mat)
    else:
        scpot = np.asarray(scpot)
        if scpot.ndim == 1:
            scpot_mat = np.tile(scpot[:, None], (1, n_energy))  # shape: (n_time, n_energy)
        else:
            scpot_mat = scpot
    energy_corr = energy_mat - scpot_mat  # shape: (n_time, n_energy)

    # Mask non-physical energies (e.g., < 0 or very low energy)
    energy_corr[energy_corr <= 10] = np.nan

    # Get cross section: shape = (n_energy,)
    cross_section = ei_cross_section(energy, species)  # cm²
    if cross_section is None:
        raise ValueError(f"No cross-section found for species: {species}")
    cross_section = np.asarray(cross_section)

    # Compute solid angle
    domega = dtheta * dphi  # steradian

    # Compute energy bin width: shape = (n_time, n_energy)
    dE = energy_corr[:, 1:] - energy_corr[:, :-1]  # shape: (n_time, n_energy - 1)
    dE_last = dE[:, -1:]                           # shape: (n_time, 1)
    dE = np.concatenate([dE, dE_last], axis=1)     # shape: (n_time, n_energy)

    # Convert DEF (eV⁻¹ cm⁻² s⁻¹ sr⁻¹) to DEF
    e_dpf = DEF / energy_corr  # shape: (n_time, n_energy)
    e_dpf = np.where(np.isnan(energy_corr), np.nan, e_dpf)

    # Total production rate: integrate over energy
    # Shape: (n_time,)
    prod_rate = np.nansum(e_dpf * cross_section[None, :] * domega * dE, axis=1)  # units: cm⁻³ s⁻¹

    return prod_rate

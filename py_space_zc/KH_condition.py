import numpy as np
from py_space_zc import dot  # assumed to be a standard vector dot-product along the last axis

def KH_condition(rho1, rho2, v1, v2, k, B1, B2):
    """
    Kelvin–Helmholtz (KH) instability criterion with an arbitrary wavevector direction.

    Parameters
    ----------
    rho1, rho2 : float or array-like
        Mass densities of region 1 and 2 in [amu·cm^-3].
        (e.g., for O+: rho = 16*n, for O2+: rho = 32*n, with n in cm^-3)
    v1, v2 : array-like (..., 3)
        Plasma bulk velocity vectors [km/s].
    k : array-like (..., 3)
        Wavevector direction (any non-zero vector). It will be normalized internally.
    B1, B2 : array-like (..., 3)
        Magnetic field vectors [nT].
    return_gamma : bool, optional
        If True, also return the growth rate gamma = sqrt(argument) [1/s];
        if False, only return the criterion argument.

    Returns
    -------
    argument : float or ndarray
        Value of the KH instability criterion in SI units:
            argument = (rho1*rho2 / (rho1+rho2)) * (ΔV·k_hat)^2
                       - ((B1·k_hat)^2 + (B2·k_hat)^2) / mu0
        If argument > 0, the KH instability can grow.
        If argument < 0, the instability is suppressed by magnetic tension.
    gamma : float or ndarray, optional
        Returned only if return_gamma=True.
        Growth rate term sqrt(argument) [1/s]. NaN if argument < 0.

    Notes
    -----
    - rho1, rho2 are given in amu·cm^-3; they are converted to SI [kg/m^3] using
      1 amu = 1.6726×10^-27 kg.
    - Velocities are given in km/s and converted to m/s.
    - Magnetic fields are given in nT and converted to Tesla.
    Author: Chi Zhang
    """

    # --- Physical constants ---
    mu0 = 4 * np.pi * 1e-7          # Vacuum permeability [H/m]
    amu_to_kg = 1.6726e-27          # Mass of 1 amu in kg

    # --- Convert inputs to numpy arrays ---
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    B1 = np.asarray(B1, dtype=float)
    B2 = np.asarray(B2, dtype=float)
    k  = np.asarray(k,  dtype=float)
    rho1 = np.asarray(rho1, dtype=float)
    rho2 = np.asarray(rho2, dtype=float)

    # --- Unit conversions ---
    v1_si = v1 * 1e3     # km/s → m/s
    v2_si = v2 * 1e3
    B1_si = B1 * 1e-9    # nT → Tesla
    B2_si = B2 * 1e-9
    rho1_si = rho1 * 1e6 * amu_to_kg   # amu·cm^-3 → kg/m^3
    rho2_si = rho2 * 1e6 * amu_to_kg


    # --- Project velocity shear and magnetic fields onto k_hat ---
    dV_si = v1_si - v2_si
    dV_par = dot(dV_si, k)   # (ΔV · k_hat)
    B1_par = dot(B1_si, k)   # (B1 · k_hat)
    B2_par = dot(B2_si, k)   # (B2 · k_hat)

    # --- KH driving vs. magnetic suppression ---
    inertia_term  = (rho1_si * rho2_si) / (rho1_si + rho2_si) **2 * (dV_par ** 2)
    magnetic_term = (B1_par ** 2 + B2_par ** 2) / mu0/ (rho1_si * rho2_si)

    argument = inertia_term - magnetic_term

    return argument

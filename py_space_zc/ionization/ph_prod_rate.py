import os
from functools import lru_cache

import numpy as np


_H = 6.626e-34
_C = 3.0e8


def _cross_section_file(species):
    species_key = species.lower().strip()
    if species_key in ("h", "p", "h+"):
        return "ph_H_cross_section.txt"
    if species_key in ("o", "o+"):
        return "ph_O_cross_section.txt"
    if species_key in ("co", "co+"):
        return "ph_CO_cross_section.txt"
    if species_key in ("co2", "co2+"):
        return "ph_CO2_cross_section.txt"
    if species_key in ("h2o", "h2o+"):
        return "ph_H2O_cross_section.txt"
    if species_key in ("n2", "n2+"):
        return "ph_N2_cross_section.txt"
    if species_key in ("na", "na+"):
        return "ph_Na_cross_section.txt"
    if species_key in ("mg", "mg+"):
        return "ph_Mg_cross_section.txt"
    if species_key in ("ca", "ca+"):
        return "ph_Ca_cross_section.txt"
    raise ValueError(f"Unsupported species for photoionization: {species!r}")


@lru_cache(maxsize=None)
def _load_cross_section(filename):
    data_path = os.path.join(os.path.dirname(__file__), filename)
    data = np.loadtxt(data_path, ndmin=2)

    lambda_nm = data[:, 0]
    sigma_cm2 = data[:, 1]
    order = np.argsort(lambda_nm)
    return lambda_nm[order], sigma_cm2[order]


def ph_cross_section(wavelength, species):
    """
    Interpolate the photoionization cross section.

    Parameters
    ----------
    wavelength : float or array-like
        Wavelength in nm.
    species : {'H', 'O', 'CO', 'CO2', 'H2O', 'N2', 'Na', 'Mg', 'Ca'}
        Neutral species.

    Returns
    -------
    sigma_cm2 : float or ndarray
        Photoionization cross section in cm^2. Values outside the tabulated
        wavelength range are set to 0, matching the MATLAB reference.
    """
    wavelength_arr = np.asarray(wavelength, dtype=float)
    scalar_input = wavelength_arr.ndim == 0

    filename = _cross_section_file(species)
    lambda_nm, sigma_cm2 = _load_cross_section(filename)

    sigma = np.interp(wavelength_arr, lambda_nm, sigma_cm2, left=0.0, right=0.0)

    if scalar_input:
        return float(np.asarray(sigma))
    return sigma


def ph_prod_rate(wavelength, W, species):
    """
    Calculate the photoionization production rate.

    This is the Python version of the MATLAB ``photo_prod_rate`` workflow.

    Parameters
    ----------
    wavelength : array-like
        EUV wavelength in nm. Its length must match the last dimension of W.
    W : array-like
        Solar spectral irradiance in W m^-2 nm^-1. This may be a 1D spectrum
        or an array whose last dimension is wavelength, for example
        ``(time, wavelength)``.
    species : {'H', 'O', 'CO', 'CO2', 'H2O', 'N2', 'Na', 'Mg', 'Ca'}
        Neutral species.

    Returns
    -------
    prod : float or ndarray
        Photoionization rate in s^-1. If W is 1D, a scalar is returned. If W
        is 2D or higher, the result has the same leading dimensions as W.

    Notes
    -----
    Unit conversion follows the MATLAB reference:

    1. Wavelength is converted from nm to m.
    2. Irradiance is converted from W m^-2 nm^-1 to W m^-3 by multiplying by
       1e9.
    3. Photon energy is E = h c / lambda.
    4. Photon flux density is phi = W / E, in photons s^-1 m^-3.
    5. Photoionization cross section is interpolated in cm^2 and converted
       to m^2.
    6. The rate is integral sigma(lambda) phi(lambda) dlambda, in s^-1.
    """
    wavelength_nm = np.asarray(wavelength, dtype=float)
    if wavelength_nm.ndim != 1:
        raise ValueError("wavelength must be a 1D array in nm.")
    if wavelength_nm.size < 2:
        raise ValueError("wavelength must contain at least two samples.")

    W_arr = np.asarray(W, dtype=float)
    if W_arr.shape[-1] != wavelength_nm.size:
        raise ValueError(
            "The last dimension of W must match the length of wavelength."
        )

    wavelength_m = wavelength_nm * 1e-9
    irradiance_w_m3 = W_arr * 1e9
    photon_energy_j = _H * _C / wavelength_m
    photon_flux = irradiance_w_m3 / photon_energy_j
    dlambda_m = np.nanmean(np.diff(wavelength_m))

    sigma_m2 = ph_cross_section(wavelength_nm, species) * 1e-4
    prod = np.nansum(sigma_m2 * photon_flux * dlambda_m, axis=-1)

    if W_arr.ndim == 1:
        return float(np.asarray(prod))
    return prod


__all__ = ["ph_cross_section", "ph_prod_rate"]

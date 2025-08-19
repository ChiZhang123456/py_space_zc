"""
Adapted from pyrfu's ts_skymap module:
(https://github.com/louis-richard/irfu-python), licensed under the MIT License.

Original code licensed under the MIT License.
Modified by Chi Zhang for compatibility with py_space_zc and to include
spacecraft velocity correction in the velocity moments calculation.
"""

import logging
import numpy as np
import xarray as xr
from scipy import constants
from pyrfu.pyrf import resample, ts_scalar, ts_tensor_xyz, ts_vec_xyz

from .match_vdf_dims import match_vdf_dims
from .convert_energy_velocity import convert_energy_velocity
from .expand_4d_grid import expand_4d_grid
from .vxyz_from_polar import vxyz_from_polar
from .get_particle_mass_charge import get_particle_mass_charge
from .plasma_moments import plasma_moments
import warnings

logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def moments_calculation(vdf, sc_pot=None, vsc_instrument=None, **kwargs):
    """
    Compute plasma moments (density, velocity, pressure, temperature, and fluxes)
    from a 4D phase space distribution function (VDF).

    Parameters
    ----------
    vdf : xarray.Dataset
        Input VDF with required variables:
        - data    : PSD [s^3/m^6], shape (n_time, n_energy, n_phi, n_theta)
        - energy  : energy levels [eV]
        - phi     : azimuthal angles [deg]
        - theta   : elevation angles [deg]
        - attrs['species'] : particle species string (e.g., 'H+', 'O2+', 'e')

    sc_pot : xarray.DataArray or None, optional
        Spacecraft potential [eV] as a time series. If None, assumed zero.

    vsc_instrument : xarray.DataArray or None, optional
        Spacecraft velocity in the instrument frame [km/s]. If None, assumed zero.

    Optional Keyword Arguments
    --------------------------
    Emin : float
        Minimum energy threshold [eV] (mask out lower energies).
    Emax : float
        Maximum energy threshold [eV] (mask out higher energies).

    Returns
    -------
    dict
        Dictionary of time series (pyrfu TSeries-like) plasma moments:
        - 'n'  : number density [cm^-3]
        - 'V'  : bulk velocity vector [km/s]
        - 'P'  : thermal pressure tensor [nPa]
        - 'P2' : total momentum flux tensor [nPa]
        - 'T'  : temperature tensor [eV]
        - 'H'  : enthalpy flux [erg/s/cm^2]
        - 'Q'  : heat flux [erg/s/cm^2]
        - 'K'  : kinetic energy flux [erg/s/cm^2]
    """

    # 1. Extract PSD and geometry from the VDF dataset
    vdf_data = vdf.data.data            # PSD [s^3/m^6]
    energy = vdf.energy.data            # energy levels [eV]
    phi = vdf.phi.data                  # azimuthal angle [deg]
    theta = vdf.theta.data              # polar angle [deg]
    particle_type = vdf.attrs["species"].lower()
    n_time = vdf_data.shape[0]

    # 2. Broadcast energy/phi/theta arrays to match PSD dimensions
    energy_new, dE_new, phi_new, theta_new = match_vdf_dims(vdf_data, energy, phi, theta)

    # 3. Expand the 4D grid (adds bin widths for phi/theta/energy)
    if "deltatheta" in vdf.attrs or "dtheta" in vdf.attrs:
        dtheta_key = "deltatheta" if "deltatheta" in vdf.attrs else "dtheta"
        deltatheta = vdf.attrs[dtheta_key]  # [deg]
        res = expand_4d_grid(vdf_data, energy_new, phi_new, theta_new,
                             particle_type, delta_theta = deltatheta)
    else:
        res = expand_4d_grid(vdf_data, energy_new, phi_new, theta_new, particle_type)

    energymat = res["energymat"]    # [eV]
    dEmat = res["dEmat"]            # [eV]
    deltaphi = res["deltaphi"]      # [deg]
    deltatheta = res["deltatheta"]  # [deg]
    phimat = res["phimat"]          # [deg]
    thetamat = res["thetamat"]      # [deg]

    # 4. Apply optional energy mask
    Emin = kwargs.get("Emin", None)
    Emax = kwargs.get("Emax", None)
    if Emin is not None and Emax is not None:
        mask = (energymat >= Emin) & (energymat <= Emax)
        energymat  = np.where(mask, energymat, np.nan)
        dEmat      = np.where(mask, dEmat, np.nan)
        phimat     = np.where(mask, phimat, np.nan)
        thetamat   = np.where(mask, thetamat, np.nan)
        deltaphi   = np.where(mask, deltaphi, np.nan)
        deltatheta = np.where(mask, deltatheta, np.nan)
        vdf_data   = np.where(mask, vdf_data, np.nan)

    # 5. Apply spacecraft potential correction
    if sc_pot is None:
        sc_pot = np.zeros(n_time)
    else:
        sc_pot = resample(sc_pot, vdf.time).data
    sc_pot = sc_pot[:, None, None, None]  # shape to match energymat
    energy_correct = energymat - sc_pot
    energy_correct = np.where(energy_correct > 0, energy_correct, np.nan)

    # 6. Prepare spacecraft velocity correction
    #    vsc_instrument is converted from km/s → m/s before passing to moments
    if vsc_instrument is None:
        vsc_instrument = np.zeros((n_time, 3), dtype=np.float64)
    else:
        vsc_instrument = resample(vsc_instrument, vdf.time).data * 1e3

    # 7. Calculate plasma moments from the corrected PSD
    moments = plasma_moments(
        vdf_data, energy_correct, dEmat,
        phimat, thetamat, deltaphi, deltatheta,
        vsc_instrument, particle_type
    )

    # 8. Package the moments into time series (pyrfu-style containers)
    output = {
        "n":  ts_scalar(vdf.time.data, moments["n"]),
        "V":  ts_vec_xyz(vdf.time.data, moments["V"]),
        "P":  ts_tensor_xyz(vdf.time.data, moments["Pressure"]),
        "P2": ts_tensor_xyz(vdf.time.data, moments["P2"]),
        "T":  ts_tensor_xyz(vdf.time.data, moments["Temp"]),
        "H":  ts_vec_xyz(vdf.time.data, moments["H"]),
        "Q":  ts_vec_xyz(vdf.time.data, moments["Q"]),
        "K":  ts_vec_xyz(vdf.time.data, moments["K"]),
    }

    return output

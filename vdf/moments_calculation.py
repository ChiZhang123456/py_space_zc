"""
Adapted from pyrfu's ts_skymap module:
(https://github.com/louis-richard/irfu-python), licensed under the MIT License

Original code licensed under the MIT License.
Modified by Chi Zhang for compatibility with py_space_zc.
"""
# Built-in imports
import logging

# 3rd party imports
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

def moments_calculation(vdf, sc_pot=None, **kwargs):
    """
    Compute plasma moments (density, velocity, pressure, temperature, fluxes)
    from a 4D phase space distribution function.

    Parameters
    ----------
    vdf : xarray.Dataset
        Input distribution function with variables:
        - data: PSD [s^3/m^6], shape (n_time, n_energy, n_phi, n_theta)
        - energy: energy levels [eV]
        - phi: azimuthal angles [deg]
        - theta: elevation angles [deg]
        - attrs['species']: particle species string

    sc_pot : xarray.DataArray or None
        Spacecraft potential [eV] as a time series (optional).

    Optional Keyword Arguments
    --------------------------
    Emin : float
        Minimum energy threshold [eV]
    Emax : float
        Maximum energy threshold [eV]

    Returns
    -------
    dict
        Dictionary of TSeries plasma moments:
        - 'n': number density [cm^-3]
        - 'V': bulk velocity vector [km/s]
        - 'P': thermal pressure tensor [nPa]
        - 'P2': total momentum flux tensor [nPa]
        - 'T': temperature tensor [eV]
        - 'H': enthalpy flux [erg/s/cm^2]
        - 'Q': heat flux [erg/s/cm^2]
        - 'K': kinetic energy flux [erg/s/cm^2]
    """
    # 1. Extract input variables
    vdf_data = vdf.data.data                               # PSD [s^3/m^6]
    energy = vdf.energy.data                               # [eV]
    phi = vdf.phi.data                                     # [deg]
    theta = vdf.theta.data                                 # [deg]
    particle_type = vdf.attrs["species"].lower()           # e.g. 'h+', 'o2+', 'e'
    n_time = vdf_data.shape[0]

    # 2. Match VDF dimension shapes (broadcast 1D to 2D)
    energy_new, dE_new, phi_new, theta_new = match_vdf_dims(vdf_data, energy, phi, theta)

    # 3. Expand to 4D VDF grid including phi/theta/energy bin widths
    if "deltatheta" in vdf.attrs or "dtheta" in vdf.attrs:
        dtheta_key = "deltatheta" if "deltatheta" in vdf.attrs else "dtheta"
        deltatheta = vdf.attrs[dtheta_key]               # [deg]
        res = expand_4d_grid(vdf_data, energy_new, phi_new, theta_new, particle_type, delta_theta = deltatheta)
    else:
        res = expand_4d_grid(vdf_data, energy_new, phi_new, theta_new, particle_type)

    energymat = res["energymat"]                          # [eV]
    dEmat = res["dEmat"]                                  # [eV]
    deltaphi = res["deltaphi"]              # [deg]
    deltatheta = res["deltatheta"]          # [deg]
    phimat = res["phimat"]                  # [deg]
    thetamat = res["thetamat"]              # [deg]


    # 4. Apply energy mask if Emin/Emax given
    Emin = kwargs.get("Emin", None)
    Emax = kwargs.get("Emax", None)
    if Emin is not None and Emax is not None:
        mask = (energymat >= Emin) & (energymat <= Emax)
        energymat   = np.where(mask, energymat, np.nan)
        dEmat       = np.where(mask, dEmat, np.nan)
        phimat      = np.where(mask, phimat, np.nan)
        thetamat    = np.where(mask, thetamat, np.nan)
        deltaphi    = np.where(mask, deltaphi, np.nan)
        deltatheta  = np.where(mask, deltatheta, np.nan)
        vdf_data    = np.where(mask, vdf_data, np.nan)
        deltaang    = np.where(mask, deltaang, np.nan)

    # 5. Apply spacecraft potential correction (if given)
    if sc_pot is None:
        sc_pot = np.zeros(n_time)
    else:
        sc_pot = resample(sc_pot, vdf.time).data
    sc_pot = sc_pot[:, None, None, None]                  # [eV]
    energy_correct = energymat - sc_pot
    energy_correct = np.where(energy_correct > 0, energy_correct, np.nan)  # Filter invalid values

    # 6. Get physical constants for this particle type
    p_mass, q_e = get_particle_mass_charge(particle_type)

    # 7. Calculate plasma moments from PSD
    moments = plasma_moments(vdf_data, energy_correct, dEmat,
                              phimat, thetamat, deltaphi, deltatheta, particle_type)

    # 8. Package results into xarray time series
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


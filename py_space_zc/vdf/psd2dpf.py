"""
Adapted from pyrfu (https://github.com/louis-richard/irfu-python),
licensed under the MIT License.

Author: Chi Zhang
Source note: this file is based on pyrfu time-series or VDF helper routines
and was modified for compatibility with py_space_zc.
"""

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import constants
from ._flux_convert_utils import broadcast_energy_to_data

def _mass_ratio(inp):
    if inp.attrs["species"].lower() in ["ions", "ion", "protons", "proton","h","h+"]:
        mass_ratio = 1
    elif inp.attrs["species"].lower() in ["alphas", "alpha", "helium", "he","he+"]:
        mass_ratio = 4
    elif inp.attrs["species"].lower() in ["electrons", "electron", "e", "e-"]:
        mass_ratio = constants.electron_mass / constants.proton_mass
    elif inp.attrs["species"].lower() in ["o", "o+"]:
        mass_ratio = 16
    elif inp.attrs["species"].lower() in ["o2", "o2+"]:
        mass_ratio = 32
    elif inp.attrs["species"].lower() in ["co2", "co2+"]:
        mass_ratio = 44
    else:
        raise ValueError("Invalid specie")

    return mass_ratio


def _convert(inp, units, mass_ratio):
    fact = 1 / (1e6 * 0.53707 * mass_ratio**2)

    if units.lower() == "s^3/cm^6":
        out = inp * 1e30 * fact
    elif units.lower() == "s^3/m^6":
        out = inp * 1e18 * fact
    elif units.lower() == "s^3/km^6":
        out = inp * fact
    else:
        raise ValueError("Invalid unit")

    return out


def psd2dpf(inp):
    r"""Compute differential particle flux from phase density.

    Parameters
    ----------
    vdf : xarray.Dataset
        Time series of the velocity distribution function with :
            * time : Time samples.
            * data : 3D velocity distribution.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    Returns
    -------
    dpf : xarray.Dataset
        Time series of the 3D differential particle flux in 1/(cm^2 s sr keV) with :
            * time : Time samples.
            * data : 3D density particle flux.
            * energy : Energy levels.
            * phi : Azimuthal angles.
            * theta : Elevation angle.

    """

    assert isinstance(inp, (xr.DataArray, xr.Dataset)), "inp must be a xarray"

    if isinstance(inp, xr.Dataset):
        tmp_data = _convert(inp.data.data, inp.data.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        energy_mat = broadcast_energy_to_data(energy, tmp_data, getattr(inp.data, "dims", None))
        tmp_data *= energy_mat
        out = inp.copy()
        out.data.data = np.squeeze(tmp_data) * 1e3
        out.data.attrs["UNITS"] = "1/(cm^2 s sr keV)"
    else:
        tmp_data = _convert(inp.data, inp.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        energy_mat = broadcast_energy_to_data(energy, tmp_data, getattr(inp, "dims", None))
        tmp_data *= energy_mat
        out = inp.copy()
        out.data = np.squeeze(tmp_data) * 1e3
        out.attrs["UNITS"] = "1/(cm^2 s sr keV)"

    return out

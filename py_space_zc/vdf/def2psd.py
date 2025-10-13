"""
Adapted from pyrfu's ts_skymap module:
(https://github.com/louis-richard/irfu-python), licensed under the MIT License

Original code licensed under the MIT License.
Modified by Chi Zhang for compatibility with py_space_zc.
"""
# Built-in imports
from typing import Union

# 3rd party imports
import numpy as np
from scipy import constants
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

def _mass_ratio(inp):
    if inp.attrs["species"].lower() in ["ions", "ion", "protons", "proton","h","h+"]:
        mass_ratio = 1
        
    elif inp.attrs["species"].lower() in ["alphas", "alpha", "helium", "he","he+"]:
        mass_ratio = 4
        
    elif inp.attrs["species"].lower() in ["electrons", "e"]:
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
    fact = 1e6 * 0.53707 * mass_ratio**2

    if units.lower() in ["kev/(cm^2 s sr kev)", "ev/(cm^2 s sr ev)", "1/(cm^2 s sr)"]:
        tmp_data = inp / 1e18 * fact
    else:
        raise ValueError("Invalid unit")

    return tmp_data


def def2psd(inp: Union[DataArray, Dataset]) -> Union[DataArray, Dataset]:
    r"""Compute phase space density from differential energy flux.

    The phase-space density is given by:

    .. math:

        f(E) = m^2 \frac{DEF}{E^2} * 0.53707,

    where :math:`m` is the particle mass in atomic mass unit, :math:`DEF` is
    the differential energy flux in 1/(cm sr s) and :math:`E` is the energy
    in eV.

    Parameters
    ----------
    inp : xarray.Dataset or xarray.DataArray
        Time series of the differential energy flux in [(cm^{2} s sr)^{-1}].

    Returns
    -------
    psd : xarray.Dataset or xarray.DataArray
        Time series of the phase space density in [s^{3} m^{-6}]

    Raises
    ------
    TypeError
        If inp is not a xarray.Dataset or xarray.DataArray.

    """
    if isinstance(inp, Dataset):
        tmp_data = _convert(inp.data.data, inp.data.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        energy_mat = np.tile(energy[:, :, None, None], (1, 1, *tmp_data.shape[2:]))
        tmp_data /= energy_mat**2
        out = inp.copy()
        out.data.data = np.squeeze(tmp_data)
        out.data.attrs["UNITS"] = "s^3/m^6"
    elif isinstance(inp, DataArray):
        tmp_data = _convert(inp.data, inp.attrs["UNITS"], _mass_ratio(inp))
        energy = inp.energy.data
        if energy.ndim == 1:
            energy_mat = np.tile(energy, (tmp_data.shape[0], 1))
        elif energy.ndim == 2:
            energy_mat = energy.copy()
        tmp_data /= energy_mat**2
        out = inp.copy()
        out.data = np.squeeze(tmp_data)
        out.attrs["UNITS"] = "s^3/m^6"
    else:
        raise TypeError("Invalid input type")

    return out

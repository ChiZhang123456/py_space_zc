# -*- coding: utf-8 -*-
"""
Crustal field model interface using spherical harmonic coefficients (SHC).

Author: Chi Zhang
Date: 2024-07-31
"""

import os
import numpy as np
import scipy.io as scio
import pyshtools as pysh
from pyrfu import pyrf
from pymagglobal.utils import i2lm_l, i2lm_m

from py_space_zc.maven import get_base_path, get_data, coords_convert, lonlat2pc, pc2lonlat
from py_space_zc.maven import get_data as maven_get_data  # optional alias
from py_space_zc import sph2cart_vec  # from py_space_zc root level

#%% 
def cf_model(alt_km, lat_deg, lon_deg):
    """
    Compute crustal magnetic field at given spherical coordinates (PC frame).

    Parameters
    ----------
    alt_km : array-like
        Altitude in km above Mars reference radius (3393.5 km).
    lat_deg : array-like
        Latitude in degrees.
    lon_deg : array-like
        Longitude in degrees.

    Returns
    -------
    Bsph_pc : ndarray
        Magnetic field in spherical coordinates (Br, Btheta, Bphi).
    Bxyz_pc : ndarray
        Magnetic field in Cartesian (x, y, z) in PC frame.
    """
    # Load SH coefficients from .mat file
    coeffs_path = os.path.join(get_base_path(), "supporting_data", "L1.mat")
    data = scio.loadmat(coeffs_path)
    coeff_array = np.array(data["L1"]).squeeze()

    # Convert to pyshtools-compatible array
    coeffs_pysh = np.zeros((2, 111, 111))  # (2, lmax+1, lmax+1)
    for i, val in enumerate(coeff_array):
        l = i2lm_l(i)
        m = abs(i2lm_m(i))
        if i2lm_m(i) < 0:
            coeffs_pysh[1, l, m] = val
        else:
            coeffs_pysh[0, l, m] = val

    # Create SHMagCoeffs object
    G110 = pysh.SHMagCoeffs.from_array(coeffs_pysh, r0=3393.5)
    Bsph_pc = G110.expand(
        a=alt_km + 3393.5, lat=lat_deg, lon=lon_deg,
        degrees=True, lmax=110, sampling=2, extend=True
    )

    # Convert to Cartesian in PC frame
    Ppc = lonlat2pc(alt_km, lon_deg, lat_deg)
    Bxyz_pc = sph2cart_vec(Ppc, Bsph_pc)

    return Bsph_pc, Bxyz_pc

#%% 
def cf_model_pc(Ppc):
    """
    Compute crustal field at given positions in PC frame.

    Parameters
    ----------
    Ppc : xarray.DataArray
        Position time series in PC coordinates.

    Returns
    -------
    dict
        {
            "Bpc_sph": spherical field [nT],
            "Bpc": Cartesian field in PC [nT],
            "Bmso": Cartesian field in MSO [nT]
        }
    """
    alt_km = np.linalg.norm(Ppc.data, axis=1) - 3393.5
    lon_deg, lat_deg = pc2lonlat(Ppc.data)

    Bsph_pc, Bxyz_pc = cf_model(alt_km, lat_deg, lon_deg)

    Bpc = pyrf.ts_vec_xyz(Ppc.time.data, Bxyz_pc, attrs={
        "name": "Crustal field",
        "unit": "nT",
        "coordinates": "PC"
    })

    Bmso = coords_convert(Bpc, "pc2mso")

    return {
        "Bpc_sph": Bsph_pc,
        "Bpc": Bpc,
        "Bmso": Bmso
    }

#%% 
def cf_model_mso(Pmso):
    """
    Compute crustal field from positions in MSO coordinates.

    Parameters
    ----------
    Pmso : xarray.DataArray
        Position time series in MSO coordinates.

    Returns
    -------
    dict
        Same as `cf_model_pc`.
    """
    Ppc = coords_convert(Pmso, "mso2pc")
    return cf_model_pc(Ppc)


#%%      
if __name__ == "__main__":
    tint = ['2015-09-18 22:21:00', '2015-09-18 22:23:30']
    B = maven_get_data(tint, 'B')  # or get_data if already imported
    Pmso = B["Pmso"]
    #Bmodel = cf_model_mso(Pmso)
















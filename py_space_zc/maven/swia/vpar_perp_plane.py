from py_space_zc import maven, vdf, time_eval
import numpy as np
import spiceypy as sp
import matplotlib.pyplot as plt
from pyrfu.pyrf import normalize

def vpar_perp_plane(swia_3d, Bmso, vg = np.linspace(-1000.0, 1000.0, 200) * 1e3):
    """
    Compute the 2D velocity distribution function (VDF) in the (v_parallel, |v_perp|) plane
    at a given time using SWIA 3D data and magnetic field vectors.

    Parameters
    ----------
    swia_3d : xarray.Dataset
        MAVEN SWIA 3D ion distribution data (should contain DEF or PSD).
    Bmso : xarray.DataArray
        Magnetic field vector in MSO coordinates [nT], dimension: (time, 3).
    vg : ndarray of float, optional
        Velocity grid for v_parallel [m/s], default: -1000 to 1000 km/s (converted to m/s).

    Returns
    -------
    f2d : xarray.DataArray
        2D reduced distribution function in (v_parallel, |v_perp|) plane at given time.
        Dimensions: (v_parallel, v_perp)
    """
    # Clear any previously loaded SPICE kernels
    sp.kclear()

    # Load MAVEN-specific SPICE kernels (position, velocity, attitude, etc.)
    maven.load_maven_spice()

    # Convert magnetic field from MSO to SWIA instrument frame, then normalize (unit vector)
    b_swia = normalize(maven.coords_convert(Bmso, 'mso2swia'))

    # Convert differential energy flux (DEF) to phase space density (PSD) [s^3/m^6]
    psd = vdf.flux_convert(swia_3d, 'def2psd')

    # Define v_parallel grid (input vg) and corresponding v_perp grid (half as many bins)
    vpar_grid = vg  # [m/s]
    n_vg = len(vg)
    max_v = np.max(vg)
    vperp_grid = np.linspace(0.0, max_v, n_vg // 2)  # [m/s], radial direction from 0 to max

    # Perform Monte Carlo integration to reduce 3D PSD to 2D (v_parallel, |v_perp|) plane
    f2d_par_perp = vdf.par_perp_reduced_dis(
        psd, b_swia,
        vpar_grid=vpar_grid,
        vperp_grid=vperp_grid,
        n_mc = 2000  # number of Monte Carlo particles
    )


    return f2d_par_perp

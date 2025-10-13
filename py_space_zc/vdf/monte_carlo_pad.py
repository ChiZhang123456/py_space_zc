#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
par_perp_reduced_dis.py

Description:
------------
This module provides functionality to project a 3D velocity distribution function (VDF)
onto a 2D (v_parallel, |v_perp|) velocity space using Monte Carlo sampling.

The key function `par_perp_reduced_dis()` performs a reduced representation of the VDF
by integrating over angular bins (phi, theta) and energy (or speed) bins, producing
a pitch-angle-resolved view of the particle phase space distribution in the magnetic
field–aligned frame.

This method is particularly useful for visualizing anisotropies, field-aligned beams,
ring distributions, and other kinetic features in plasma data.

The method handles:
- Arbitrary spacecraft frames
- Relativistic speed calculation from energy
- Species-based mass identification
- Spacecraft potential correction
- Lower-energy thresholding
- Optional bin weighting (linear/log) for adaptive sampling

It is tailored for use with space plasma datasets such as MAVEN/SWIA, Tianwen-1, MMS/FPI, etc.,
and supports both xarray-native structures and Python numerical operations with Numba acceleration.

Functions:
----------
- par_perp_reduced_dis : Main function to reduce 3D VDF to 2D (v_parallel, |v_perp|)
- _sph_dist : Wrapper to call Numba-based integrator per time step
- monte_carlo_pad : Numba-accelerated Monte Carlo integrator
- _make_edges : Utility to construct bin edges from grid centers

Typical Usage:
--------------
    from py_space_zc import maven
    B, swia_3d = maven.load_data(tint, ['B','swia_3d'])
    out = par_perp_reduced_dis(swia_3d, B)
    ax, _, _ = plot.plot_pcolor(None, out.vpar, out.vperp, out[0].T, cscale="log")

Author: Chi Zhang (Boston University, Center for Space Physics)
Date: September 2025
License: MIT-style (open source, reusable)
"""


# Third party imports
import numpy as np
import numba
import xarray as xr
import tqdm
from scipy.constants import electron_mass, electron_volt, proton_mass, speed_of_light

# Local imports
from pyrfu.pyrf.datetime642iso8601 import datetime642iso8601
from pyrfu.pyrf.resample import resample
from pyrfu.pyrf.time_clip import time_clip
from pyrfu.pyrf.ts_scalar import ts_scalar


@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def monte_carlo_pad(
    vdf, v, phi, theta,
    d_v, d_v_m, d_phi, d_theta,
    vpar_edges, vperp_edges,
    d_a_grid,
    v_lim, a_lim,
    n_mc, b_vec
):
    """
    Monte Carlo interpolation of 3D VDF onto a 2D (v_parallel, |v_perp|) grid
    using cylindrical coordinates (vz, sqrt(vx^2 + vy^2)).
    Parameters:
    -----------
    vdf : ndarray, shape (n_v, n_phi, n_theta)
    3D distribution function in spherical velocity coordinates
    v, phi, theta : ndarrays
    Speed, azimuthal, and polar angle grids
    d_v, d_v_m, d_phi, d_theta : ndarrays
    Bin widths for v, midpoints for dv, phi, and theta
    vpar_edges, vperp_edges : ndarrays
    1D grid edges for v_parallel and |v_perp|
    d_a_grid : ndarray, shape (n_perp, n_par)
    Grid cell areas in (vperp, vpar) space
    v_lim : list or ndarray
    Allowed v_parallel range [v_min, v_max]
    a_lim : list or ndarray
    Allowed pitch angle range [alpha_min, alpha_max] in radians
    n_mc : ndarray
    Number of Monte Carlo particles for each grid cell
    b_vec : ndarray, shape (3,)
    Magnetic field vector (used to define v_parallel direction)
    """
    n_v, n_phi, n_theta = vdf.shape
    n_par = len(vpar_edges) - 1
    n_perp = len(vperp_edges) - 1
    f_g = np.zeros((n_par, n_perp))

    # Normalize b vector (used only for determining v_parallel = vz)
    b_norm = np.sqrt(b_vec[0]**2 + b_vec[1]**2 + b_vec[2]**2)
    if b_norm == 0.0:
        return f_g
    bx, by, bz = b_vec[0] / b_norm, b_vec[1] / b_norm, b_vec[2] / b_norm

    for i in numba.prange(n_v):
        for j in range(n_phi):
            for k in range(n_theta):
                f_ijk = vdf[i, j, k]
                if f_ijk == 0.0:
                    continue

                n_mc_ijk = n_mc[i, j, k]
                if n_mc_ijk <= 0:
                    continue

                th = theta[k] if theta.ndim == 1 else theta[i, k]
                dth = d_theta[k] if d_theta.ndim == 1 else d_theta[i, k]
                dtau_ijk = v[i]**2 * np.cos(th) * d_v[i] * d_phi[j] * dth
                c_ijk = dtau_ijk / n_mc_ijk

                for _ in range(n_mc_ijk):
                    v_mc = v[i] - np.random.random() * d_v[i] - d_v_m[0]
                    phi_mc = phi[j] + (np.random.random() - 0.5) * d_phi[j]
                    theta_mc = th + (np.random.random() - 0.5) * dth

                    vx = v_mc * np.cos(theta_mc) * np.cos(phi_mc)
                    vy = v_mc * np.cos(theta_mc) * np.sin(phi_mc)
                    vz = v_mc * np.sin(theta_mc)

                    vpar = vx * bx + vy * by + vz * bz
                    vperp = np.sqrt(v_mc**2 - vpar**2)

                    alpha = np.arcsin(vpar / v_mc)
                    if not ((vpar >= v_lim[0]) and (vpar < v_lim[1]) and
                            (alpha >= a_lim[0]) and (alpha < a_lim[1])):
                        continue

                    i_par = np.searchsorted(vpar_edges, vpar) - 1
                    j_perp = np.searchsorted(vperp_edges, vperp) - 1

                    if (0 <= i_par < n_par) and (0 <= j_perp < n_perp):
                        f_g[i_par, j_perp] += f_ijk * c_ijk / d_a_grid[j_perp, i_par]

    return f_g



def _sph_dist(vdf, speed, phi, theta, vpar_grid, vperp_grid, b_vec, **kwargs):
    """Wrapper to integrate VDF to (v_par, |v_perp|) plane."""
    n_mc = kwargs.get("n_mc", 10)
    weight = kwargs.get("weight", None)

    v_lim = np.array(kwargs.get("v_lim", [-np.inf, np.inf]), dtype=np.float64)
    a_lim = np.deg2rad(np.array(kwargs.get("a_lim", [-180.0, 180.0]), dtype=np.float64))

    d_phi = np.abs(np.median(np.diff(phi))) * np.ones_like(phi)

    if theta.ndim == 1:
        d_theta = np.abs(np.median(np.diff(np.sort(theta)))) * np.ones_like(theta)
    elif theta.ndim == 2:
        d_theta = np.zeros_like(theta)
        for i in range(theta.shape[0]):
            d_theta[i, :] = np.abs(np.median(np.diff(np.sort(theta[i, :]))))

    d_v = np.gradient(speed)
    d_v_m = np.gradient(speed) / 2.0

    # Build grid edges
    def make_edges(grid):
        edges = np.zeros(len(grid) + 1)
        edges[1:-1] = grid[:-1] + np.diff(grid) / 2.0
        edges[0] = grid[0] - np.diff(grid[:2]).item() / 2.0
        edges[-1] = grid[-1] + np.diff(grid[-2:]).item() / 2.0
        return edges

    vpar_edges = make_edges(vpar_grid)
    vperp_edges = make_edges(vperp_grid)

    # Area element in cylindrical coordinates: 2π * vperp * dvperp * dvpar
    vperp_mid = 0.5 * (vperp_edges[:-1] + vperp_edges[1:])
    dvperp = np.diff(vperp_edges)
    dvpar = np.diff(vpar_edges)
    d_a_grid = 2 * np.pi * np.outer(vperp_mid, dvpar)  # shape (n_vperp, n_vpar)
    # da = 2πv⊥ * dv⊥ *dv∥



    n_sum = n_mc * np.sum(vdf != 0)
    if weight == "lin":
        n_mc_mat = np.ceil(n_sum / np.sum(vdf) * vdf)
    elif weight == "log":
        n_mc_mat = np.ceil(n_sum / np.sum(np.log10(vdf + 1)) * np.log10(vdf + 1))
    else:
        n_mc_mat = np.zeros_like(vdf)
        n_mc_mat[vdf != 0] = n_mc

    n_mc_mat = n_mc_mat.astype(int)

    f_g = monte_carlo_pad(vdf, speed, phi, theta,
                          d_v, d_v_m, d_phi, d_theta,
                          vpar_edges, vperp_edges,
                          d_a_grid, v_lim, a_lim, n_mc_mat, b_vec)

    return {
        "f": f_g,
        "vpar": vpar_grid,
        "vperp": vperp_grid,
        "vpar_edges": vpar_edges,
        "vperp_edges": vperp_edges,
    }

def par_perp_reduced_dis(vdf, bdmpa,
                     vpar_grid = np.linspace(-1000.0, 1000.0, 200) * 1e3,
                     vperp_grid  = np.linspace(0.0, 1000.0, 100) * 1e3,
                     **kwargs):
    """
    Monte Carlo method to project 3D VDF to 2D (v_parallel, |v_perp|).

    Parameters
    ----------
    vdf : xarray.Dataset
        3D distribution function with keys: data, energy, phi, theta, time.
    bdmpa : xarray.DataArray
        Magnetic field vector in the same frame as VDF (e.g., DMPA).
    vpar_grid : ndarray
        1D array defining the grid of v_parallel (can be symmetric).
    vperp_grid : ndarray
        1D array defining the grid of |v_perp| (must be >= 0).
    **kwargs :
        Optional parameters:
        - n_mc: number of Monte Carlo particles (default 100)
        - weight: "lin", "log", or None
        - tint: [start, end] time interval
        - sc_pot: spacecraft potential (xarray.DataArray or float)
        - lower_e_lim: threshold energy to exclude low-energy bins
        - v_lim: [v_min, v_max] range of v_parallel
        - a_lim: [angle_min, angle_max] in degrees (pitch angle)

    Returns
    -------
    xr.DataArray
        Reduced distribution function [time, vpar, vperp] with units s^3/m^6.
    """

    vdf_time = vdf.time
    tint = kwargs.get("tint", list(datetime642iso8601(vdf_time.data[[0, -1]])))
    vdf_time = time_clip(vdf_time, tint)
    vdf_energy = time_clip(vdf.energy, tint).copy()
    vdf_phi = time_clip(vdf.phi, tint).copy()
    vdf_theta = vdf.theta.copy()
    vdf_data = time_clip(vdf.data, tint).copy()

    # Convert units
    unit = vdf.data.attrs.get("UNITS", "s^3/cm^6").lower()
    if unit == "s^3/cm^6":
        vdf_data *= 1e12
    elif unit == "s^3/km^6":
        vdf_data *= 1e-18
    elif unit != "s^3/m^6":
        raise ValueError("Invalid units in VDF")

    n_t, n_en, _, _ = vdf_data.shape
    n_vpar, n_vperp = len(vpar_grid), len(vperp_grid)
    f_g = np.zeros([n_t, n_vpar, n_vperp])
    bdmpa = resample(bdmpa, vdf).data

    # Optional kwargs
    n_mc = kwargs.get("n_mc", 100)
    weight = kwargs.get("weight", None)
    v_lim = kwargs.get("v_lim", [-np.inf, np.inf])
    a_lim = kwargs.get("a_lim", [-180.0, 180.0])

    sc_pot = kwargs.get("sc_pot", ts_scalar(vdf_time.data, np.zeros(n_t)))
    sc_pot = resample(sc_pot, vdf).data

    lower_e_lim = kwargs.get("lower_e_lim", 0.0)
    if isinstance(lower_e_lim, xr.DataArray):
        lower_e_lim = resample(lower_e_lim, vdf_time).data
    elif isinstance(lower_e_lim, float):
        lower_e_lim = np.tile(lower_e_lim, n_t)
    else:
        raise TypeError("Invalid lower_e_lim")

    # Mass by species
    if vdf.species.lower() in ["electrons", "e", "e-","electron"]:
        m_p = electron_mass
    elif vdf.species.lower() in ["ions", "h", "h+"]:
        m_p = proton_mass
    elif vdf.species.lower() in ["he", "he+", "he++"]:
        m_p = 4 * proton_mass
    elif vdf.species.lower() in ["o", "o+"]:
        m_p = 16 * proton_mass
    elif vdf.species.lower() in ["o2", "o2+"]:
        m_p = 32 * proton_mass
    elif vdf.species.lower() in ["co2", "co2+"]:
        m_p = 44 * proton_mass
    else:
        raise ValueError(f"Invalid species: {vdf.species}")

    for i_t in range(n_t):
        f_3d = np.squeeze(vdf_data.data[i_t, ...]).astype(np.float64)
        energy = vdf_energy.data[i_t, :].astype(np.float64)
        bvec = np.squeeze(bdmpa[i_t, :])

        # Remove low energy bins
        e_min = max(lower_e_lim[i_t], sc_pot[i_t])
        f_3d[energy < e_min] = 0.0
        energy -= sc_pot[i_t]
        energy[energy < 0] = 0.0

        # Convert E → v (relativistic)
        gamma = 1 + electron_volt * energy / (m_p * speed_of_light**2)
        speed = speed_of_light * np.sqrt(1 - 1 / gamma**2)

        phi = np.deg2rad(vdf_phi.data[i_t, :].astype(np.float64) - 180.0)
        theta = vdf_theta.data.astype(np.float64)
        if theta.ndim == 2 and theta.shape[0] == n_t:
            theta = theta[i_t, :]
        elif theta.ndim == 3:
            theta = theta[i_t, :, :]
        theta = np.deg2rad(theta - 90.0)

        options = dict(n_mc=n_mc, weight=weight, v_lim=v_lim, a_lim=a_lim)
        tmp = _sph_dist(f_3d, speed, phi, theta, vpar_grid, vperp_grid, bvec, **options)
        f_g[i_t, ...] = tmp["f"]

    coords = [vdf_time.data, vpar_grid / 1e3, vperp_grid / 1e3]  # km/s
    dims = ["time", "vpar", "vperp"]
    return xr.DataArray(f_g, coords=coords, dims=dims)



if __name__ == "__main__":
    from py_space_zc import maven, vdf, plot
    from pyrfu.pyrf import normalize
    import spiceypy as sp
    import matplotlib.pyplot as plt

    sp.kclear()
    maven.load_maven_spice()
    tint = ["2018-10-18T20:10:00", "2018-10-18T20:15:00"]
    B, swia_3d = maven.load_data(tint, ['B','swia_3d'])
    bswia = normalize(maven.coords_convert(B['Bmso'], 'mso2swia'))
    psd = vdf.flux_convert(swia_3d, 'def2psd')
    out = par_perp_reduced_dis(psd, bswia,
                               vpar_grid = np.linspace(-1500.0, 1500.0, 300) * 1e3,
                               vperp_grid  = np.linspace(0.0, 1500.0, 150) * 1e3,
                               n_mc = 2000)
    data = np.squeeze(out.data[10,:,:])
    vpar = out.vpar.data
    vperp = out.vperp.data
    ax,_,_=plot.plot_pcolor(None, vpar, vperp, data.T, cscale="log")
    ax.set_aspect("equal")
    plt.show()



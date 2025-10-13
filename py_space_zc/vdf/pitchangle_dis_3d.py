#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

# Built-in imports
import warnings

# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from pyrfu.pyrf import resample, normalize, time_clip

from .flux_convert import flux_convert
from .match_vdf_dims import match_vdf_dims
from .convert_energy_velocity import convert_energy_velocity
from .expand_4d_grid import expand_4d_grid
from .vxyz_from_polar import vxyz_from_polar
from .get_particle_mass_charge import get_particle_mass_charge
from .pitchangle_dis import pitchangle_dis
from .pitchangle_merge_energy import pitchangle_merge_energy


logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def pitchangle_dis_3d(vdf, b_xyz, delta_angles = 22.5):
    r"""Computes the pitch angle distributions from l1b brst particle data.

    Parameters
    ----------
    vdf : xarray.Dataset
        to fill
    b_xyz : xarray.DataArray
        to fill
    tint : list of str, Optional
        Time interval for closeup.

    Returns
    -------
    pad : xarray.DataArray
        Particle pitch angle distribution

    Other Parameters
    ----------------
    angles : int or float or list of ndarray
        User defined angles.

    Examples

    """

    # Default pitch angles. 22.5 degree angle widths
    angles_v = np.linspace(delta_angles, 180, int(180 / delta_angles))
    d_angles = np.median(np.diff(angles_v)) * np.ones(len(angles_v))
    pitch_angles = angles_v - d_angles / 2
    n_angles = len(angles_v)


    # 1. Extract PSD and geometry from the VDF dataset
    time = vdf.time.data
    vdf_data = vdf.data.data            # PSD [s^3/m^6]
    energy = vdf.energy.data            # energy levels [eV]
    phi = vdf.phi.data                  # azimuthal angle [deg]
    theta = vdf.theta.data              # polar angle [deg]
    particle_type = vdf.attrs["species"].lower()

    # 2. Broadcast energy/phi/theta arrays to match PSD dimensions
    energy_new, dE_new, phi_new, theta_new = match_vdf_dims(vdf_data, energy, phi, theta)
    res = expand_4d_grid(vdf_data, energy_new, phi_new, theta_new, particle_type,)
    Vx_mat, Vy_mat, Vz_mat = vxyz_from_polar(res["energymat"], res["phimat"], res["thetamat"], particle_type)
    Vt_mat = np.sqrt(Vx_mat**2 + Vy_mat**2 + Vz_mat**2)
    vx_mat = -Vx_mat / Vt_mat  # notice it is reversed direction
    vy_mat = -Vy_mat / Vt_mat
    vz_mat = -Vz_mat / Vt_mat

    b_xyz = resample(b_xyz, vdf)
    b_vec = normalize(b_xyz)

    n_time, n_energy, n_phi, n_theta = vdf_data.shape
    b_vec_x = np.transpose(np.tile(b_vec.data[:, 0], [n_energy, n_phi, n_theta, 1]),[3, 0, 1, 2],)
    b_vec_y = np.transpose(np.tile(b_vec.data[:, 1], [n_energy, n_phi, n_theta, 1]),[3, 0, 1, 2],)
    b_vec_z = np.transpose(np.tile(b_vec.data[:, 2], [n_energy, n_phi, n_theta, 1]),[3, 0, 1, 2],)

    theta_b = np.rad2deg(
        np.arccos(
            vx_mat * np.squeeze(b_vec_x)
            + vy_mat * np.squeeze(b_vec_y)
            + vz_mat * np.squeeze(b_vec_z),),)

    pitchangle, pad_arr = pitchangle_dis(vdf_data, theta_b, delta_angles=delta_angles)

    if energy.ndim == 2:
        pad = xr.Dataset(
            {
                "data": (["time", "idx0", "idx1"], pad_arr,),
                "energy": (["time", "idx0"], energy,),
                "pitchangle": (["idx1"], pitchangle),
                "time": time,
                "idx0": np.arange(n_energy),
                "idx1": np.arange(len(pitchangle)),
            },
        )
    elif energy.ndim == 1:
        pad = xr.Dataset(
            {
                "data": (["time", "idx0", "idx1"], pad_arr,),
                "energy": (["time", "idx0"], np.tile(energy, (n_time, 1))),
                "pitchangle": (["idx1"], pitchangle),
                "time": time,
                "idx0": np.arange(n_energy),
                "idx1": np.arange(len(pitchangle)),
            },
        )


    pad.attrs = vdf.attrs
    pad.attrs["delta_pitchangle_minus"] = d_angles * 0.5
    pad.attrs["delta_pitchangle_plus"] = d_angles * 0.5

    pad.time.attrs = vdf.time.attrs
    pad.energy.attrs = vdf.energy.attrs
    pad.data.attrs["UNITS"] = vdf.data.attrs["UNITS"]

    return pad
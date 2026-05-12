#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gyrophase and pitch angle binning for skymap VDFs."""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyrfu.pyrf import normalize, resample

from ..ts_skymap import ts_skymap
from .vxyz_from_polar import vxyz_from_polar


def _bin_centers(width: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open bin edges and centers for [0, upper)."""
    if width <= 0:
        raise ValueError("Bin width must be positive.")

    n_bins_float = upper / width
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins_float, n_bins):
        raise ValueError(f"Bin width {width} must divide {upper} exactly.")

    edges = np.linspace(0.0, upper, n_bins + 1)
    centers = edges[:-1] + 0.5 * width
    return edges, centers


def _field_aligned_basis(ez: np.ndarray, gyro_reference: str = "global_x"):
    """Build a deterministic right-handed basis with z along B0."""
    if gyro_reference == "global_x":
        ref_primary = np.array([1.0, 0.0, 0.0])
        ref_secondary = np.array([0.0, 1.0, 0.0])
    elif gyro_reference == "global_y":
        ref_primary = np.array([0.0, 1.0, 0.0])
        ref_secondary = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError("gyro_reference must be 'global_x' or 'global_y'.")

    ref = np.tile(ref_primary, (ez.shape[0], 1))
    parallel = np.abs(np.sum(ref * ez, axis=1)) > 0.98
    ref[parallel] = ref_secondary

    # Handedness: ex = ref x ez, ey = ez x ex, so ex x ey = ez.
    ex = np.cross(ref, ez)
    ex_norm = np.linalg.norm(ex, axis=1)
    valid = np.isfinite(ex_norm) & (ex_norm > 0.0) & np.all(np.isfinite(ez), axis=1)
    ex[valid] = ex[valid] / ex_norm[valid, None]
    ex[~valid] = np.nan

    ey = np.cross(ez, ex)
    return ex, ey, valid


def _match_vdf_dims_no_de(data, energy, phi, theta):
    """Broadcast skymap coordinates without requiring energy bin widths."""
    n_time, n_energy, n_phi, n_theta = data.shape

    if energy.ndim == 1 and energy.size == n_energy:
        energy_new = np.tile(energy[None, :], (n_time, 1))
    elif energy.shape == (n_time, n_energy):
        energy_new = energy
    else:
        raise ValueError(f"Incorrect dimensions for energy input: {energy.shape}")

    if phi.ndim == 1 and phi.size == n_phi:
        phi_new = np.tile(phi[None, :], (n_time, 1))
    elif phi.shape == (n_time, n_phi):
        phi_new = phi
    else:
        raise ValueError(f"Incorrect dimensions for phi input: {phi.shape}")

    if theta.ndim == 1 and theta.size == n_theta:
        theta_new = np.tile(theta[None, None, :], (n_time, n_energy, 1))
    elif theta.shape == (n_time, n_theta):
        theta_new = np.tile(theta[:, None, :], (1, n_energy, 1))
    elif theta.shape == (n_energy, n_theta):
        theta_new = np.tile(theta[None, :, :], (n_time, 1, 1))
    elif theta.shape == (n_time, n_energy, n_theta):
        theta_new = theta
    else:
        raise ValueError(f"Incorrect dimensions for theta input: {theta.shape}")

    return energy_new, phi_new, theta_new


def gyrophase_pitchangle_dis(
    vdf_in: xr.Dataset,
    b_xyz: xr.DataArray,
    delta_pitchangle: float = 30.0,
    delta_gyrophase: float = 45.0,
    direction_is_velocity: bool = True,
    gyro_reference: str = "global_x",
    statistic: str = "sum",
) -> xr.Dataset:
    """Rebin a skymap VDF into energy, gyrophase, and pitch angle.

    Parameters
    ----------
    vdf_in : xarray.Dataset
        A ``py_space_zc.ts_skymap`` style VDF with ``data`` dimensions
        ``(time, idx0, idx1, idx2)`` for time, energy, phi, and theta.
        Energy is in eV. Phi and theta are in degrees.
    b_xyz : xarray.DataArray
        Background magnetic field in the same instrument coordinate system,
        in ``ts_vec_xyz`` format with dimensions ``(time, comp)``.
    delta_pitchangle, delta_gyrophase : float
        Bin widths in degrees. They must divide 180 and 360 exactly.
    direction_is_velocity : bool
        If True, use the velocity direction convention from
        ``py_space_zc.vdf.pitchangle_dis_3d``, namely ``-V/|V|`` after
        converting skymap angles to Cartesian velocity. If False, use the
        arrival direction ``+V/|V|``.
    gyro_reference : {"global_x", "global_y"}
        Reference axis used to define the field-aligned perpendicular basis.
        The basis is ``ez = B0/|B0|``, ``ex = ref x ez``, and
        ``ey = ez x ex``. Gyrophase is atan2(vperp dot ey, vperp dot ex),
        wrapped to [0, 360) degrees.
    statistic : {"sum", "mean"}
        How to combine original phi and theta bins inside each output bin.
        Empty bins are filled with NaN.

    Returns
    -------
    xarray.Dataset
        A ``ts_skymap`` style dataset. The original ``phi`` coordinate is
        replaced by gyrophase centers, and the original ``theta`` coordinate
        is replaced by pitch angle centers. The distribution has dimensions
        ``(time, energy, gyrophase, pitchangle)``.
    """
    if statistic not in {"sum", "mean"}:
        raise ValueError("statistic must be 'sum' or 'mean'.")

    time = vdf_in.time.data
    data = np.asarray(vdf_in.data.data, dtype=float)
    energy = np.asarray(vdf_in.energy.data)
    phi = np.asarray(vdf_in.phi.data)
    theta = np.asarray(vdf_in.theta.data)
    species = vdf_in.attrs.get("species", vdf_in.data.attrs.get("species", "H+"))

    n_time, n_energy, _, _ = data.shape
    energy_new, phi_new, theta_new = _match_vdf_dims_no_de(data, energy, phi, theta)
    grid = {
        "energymat": np.tile(
            energy_new[:, :, None, None], (1, 1, data.shape[2], data.shape[3])
        ),
        "phimat": np.tile(
            phi_new[:, None, :, None], (1, data.shape[1], 1, data.shape[3])
        ),
        "thetamat": np.tile(theta_new[:, :, None, :], (1, 1, data.shape[2], 1)),
    }
    vx, vy, vz = vxyz_from_polar(
        grid["energymat"], grid["phimat"], grid["thetamat"], species.lower()
    )

    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sign = -1.0 if direction_is_velocity else 1.0
        vhat = sign * np.stack((vx / speed, vy / speed, vz / speed), axis=-1)

    b_resampled = normalize(resample(b_xyz, vdf_in))
    ez = np.asarray(b_resampled.data, dtype=float)
    ex, ey, valid_b = _field_aligned_basis(ez, gyro_reference=gyro_reference)

    mu = np.sum(vhat * ez[:, None, None, None, :], axis=-1)
    mu = np.clip(mu, -1.0, 1.0)
    pitch = np.rad2deg(np.arccos(mu))

    vpara = mu[..., None] * ez[:, None, None, None, :]
    vperp = vhat - vpara
    gyro_x = np.sum(vperp * ex[:, None, None, None, :], axis=-1)
    gyro_y = np.sum(vperp * ey[:, None, None, None, :], axis=-1)
    gyro = np.mod(np.rad2deg(np.arctan2(gyro_y, gyro_x)), 360.0)
    gyro[np.linalg.norm(vperp, axis=-1) < 1e-12] = 0.0

    invalid_time = ~valid_b
    if np.any(invalid_time):
        pitch[invalid_time, ...] = np.nan
        gyro[invalid_time, ...] = np.nan

    pitch_edges, pitch_centers = _bin_centers(delta_pitchangle, 180.0)
    gyro_edges, gyro_centers = _bin_centers(delta_gyrophase, 360.0)

    out = np.full((n_time, n_energy, len(gyro_centers), len(pitch_centers)), np.nan)
    finite_data = np.isfinite(data)
    finite_coord = np.isfinite(pitch) & np.isfinite(gyro)

    for i_g in range(len(gyro_centers)):
        g0, g1 = gyro_edges[i_g], gyro_edges[i_g + 1]
        gyro_mask = (gyro >= g0) & (gyro < g1)
        for i_p in range(len(pitch_centers)):
            p0, p1 = pitch_edges[i_p], pitch_edges[i_p + 1]
            if i_p == len(pitch_centers) - 1:
                pitch_mask = (pitch >= p0) & (pitch <= p1)
            else:
                pitch_mask = (pitch >= p0) & (pitch < p1)
            mask = gyro_mask & pitch_mask & finite_coord & finite_data
            count = np.sum(mask, axis=(2, 3))
            values = np.where(mask, data, np.nan)
            summed = np.nansum(values, axis=(2, 3))
            if statistic == "sum":
                reduced = summed
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    reduced = summed / count
            reduced[count == 0] = np.nan
            out[:, :, i_g, i_p] = reduced

    attrs = dict(vdf_in.data.attrs)
    if "UNITS" not in attrs and "UNITS" in vdf_in.attrs:
        attrs["UNITS"] = vdf_in.attrs["UNITS"]

    glob_attrs = dict(vdf_in.attrs)
    glob_attrs["delta_pitchangle_minus"] = np.full_like(pitch_centers, delta_pitchangle / 2)
    glob_attrs["delta_pitchangle_plus"] = np.full_like(pitch_centers, delta_pitchangle / 2)
    glob_attrs["delta_gyrophase_minus"] = np.full_like(gyro_centers, delta_gyrophase / 2)
    glob_attrs["delta_gyrophase_plus"] = np.full_like(gyro_centers, delta_gyrophase / 2)
    glob_attrs["gyro_basis"] = f"ez=B0/|B0|, ex={gyro_reference} x ez, ey=ez x ex"
    glob_attrs["gyro_direction"] = "atan2(vperp dot ey, vperp dot ex), wrapped to [0, 360)"
    glob_attrs["direction_is_velocity"] = direction_is_velocity
    glob_attrs["bin_statistic"] = statistic

    coords_attrs = {}
    for coord in ("time", "energy"):
        if coord in vdf_in:
            coords_attrs[coord] = dict(vdf_in[coord].attrs)
    coords_attrs["phi"] = {"UNITS": "degrees", "FIELDNAM": "gyrophase"}
    coords_attrs["theta"] = {"UNITS": "degrees", "FIELDNAM": "pitchangle"}

    result = ts_skymap(
        time,
        out,
        energy_new,
        gyro_centers,
        pitch_centers,
        attrs=attrs,
        glob_attrs=glob_attrs,
        coords_attrs=coords_attrs,
    )
    result = result.rename({"phi": "gyrophase", "theta": "pitchangle"})
    result.gyrophase.attrs.update(coords_attrs["phi"])
    result.pitchangle.attrs.update(coords_attrs["theta"])
    return result


gyrophase_pitchangle_dis_3d = gyrophase_pitchangle_dis

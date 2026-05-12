#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit adiabatic Fermi and betatron electron energy changes.

The model follows the standard guiding-center mapping used for pitch-angle
distribution studies:

    E_parallel,1 = Ff * E_parallel,0
    E_perp,1 = Fb * E_perp,0

where Ff is the Fermi factor and Fb is the betatron factor.  Phase space
density is assumed to be conserved along the mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares


def def_to_psd_shape(def_data, energy, energy_axis=-2):
    """Convert differential energy flux to a PSD-proportional quantity.

    For non-relativistic electrons, phase space density is proportional to
    differential energy flux divided by E**2. The omitted constant does not
    affect the fitted Fermi or betatron factors when source and consequence
    use the same instrument units.
    """
    data = np.asarray(def_data, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if np.any(energy <= 0):
        raise ValueError("energy must be positive.")

    shape = [1] * data.ndim
    shape[energy_axis] = energy.size
    return data / energy.reshape(shape) ** 2


def _coord_from_xarray(data, energy, pitch_angle):
    if energy is None:
        for name in ("energy", "energy_eV", "e", "E"):
            if name in data.coords:
                energy = data.coords[name].data
                break
    if pitch_angle is None:
        for name in ("pitch_angle", "pitchangle", "pa", "theta", "alpha"):
            if name in data.coords:
                pitch_angle = data.coords[name].data
                break
    return energy, pitch_angle


def _to_energy_pitch_array(psd, energy=None, pitch_angle=None):
    """Return a 2-D array with dimensions (energy, pitch_angle)."""
    if hasattr(psd, "dims") and hasattr(psd, "coords"):
        energy, pitch_angle = _coord_from_xarray(psd, energy, pitch_angle)

        dims = list(psd.dims)
        energy_dim = None
        pitch_dim = None
        for dim in dims:
            if dim in ("energy", "energy_eV", "e", "E"):
                energy_dim = dim
            if dim in ("pitch_angle", "pitchangle", "pa", "theta", "alpha"):
                pitch_dim = dim

        if "time" in dims:
            psd = psd.mean(dim="time", skipna=True)
            dims = list(psd.dims)

        if energy_dim is None or pitch_dim is None:
            if psd.ndim != 2:
                raise ValueError("xarray input must contain energy and pitch-angle dimensions.")
            arr = np.asarray(psd.data, dtype=float)
        else:
            psd = psd.transpose(energy_dim, pitch_dim)
            arr = np.asarray(psd.data, dtype=float)
    else:
        arr = np.asarray(psd, dtype=float)

    if arr.ndim != 2:
        raise ValueError("psd must be a 2-D energy by pitch-angle array, or an xarray DataArray with optional time.")
    if energy is None or pitch_angle is None:
        raise ValueError("energy and pitch_angle must be supplied unless they are xarray coordinates.")

    energy = np.asarray(energy, dtype=float).squeeze()
    pitch_angle = np.asarray(pitch_angle, dtype=float).squeeze()
    if energy.ndim != 1 or pitch_angle.ndim != 1:
        raise ValueError("energy and pitch_angle must be one dimensional.")
    if arr.shape != (energy.size, pitch_angle.size):
        if arr.shape == (pitch_angle.size, energy.size):
            arr = arr.T
        else:
            raise ValueError("psd shape must match (len(energy), len(pitch_angle)).")

    order_e = np.argsort(energy)
    order_a = np.argsort(pitch_angle)
    return arr[np.ix_(order_e, order_a)], energy[order_e], pitch_angle[order_a]


def _inverse_energy_pitch(energy, pitch_angle, fermi_factor, betatron_factor):
    if fermi_factor <= 0 or betatron_factor <= 0:
        raise ValueError("fermi_factor and betatron_factor must be positive.")

    energy = np.asarray(energy, dtype=float)
    pitch_angle = np.asarray(pitch_angle, dtype=float)
    ee, aa = np.meshgrid(energy, pitch_angle, indexing="ij")

    alpha = np.deg2rad(aa)
    cos2 = np.cos(alpha) ** 2
    sin2 = np.sin(alpha) ** 2

    e0_parallel = ee * cos2 / fermi_factor
    e0_perp = ee * sin2 / betatron_factor
    e0 = e0_parallel + e0_perp

    with np.errstate(invalid="ignore", divide="ignore"):
        cos0_abs = np.sqrt(np.clip(e0_parallel / e0, 0.0, 1.0))
    alpha0_small = np.rad2deg(np.arccos(cos0_abs))
    alpha0 = np.where(aa <= 90.0, alpha0_small, 180.0 - alpha0_small)

    return e0, alpha0


def _make_interpolator(psd, energy, pitch_angle, log_interp=True, fill_value=np.nan):
    values = np.asarray(psd, dtype=float)
    if log_interp:
        values = np.where(values > 0, np.log10(values), np.nan)

    return RegularGridInterpolator(
        (energy, pitch_angle),
        values,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )


def fermi_betatron_map(
    source_psd,
    energy=None,
    pitch_angle=None,
    fermi_factor=1.0,
    betatron_factor=1.0,
    target_energy=None,
    target_pitch_angle=None,
    log_interp=True,
    fill_value=np.nan,
    return_source_grid=False,
):
    """Map a source electron PSD to a consequence PSD.

    Parameters
    ----------
    source_psd : array_like or xarray.DataArray
        Source PSD with shape (energy, pitch_angle). If a time dimension is
        present in an xarray input, it is averaged first.
    energy, pitch_angle : array_like
        Source energy in eV and pitch angle in degrees.
    fermi_factor, betatron_factor : float
        Energy-change factors for parallel and perpendicular energies.
        Values larger than 1 indicate acceleration, values smaller than 1
        indicate cooling.
    target_energy, target_pitch_angle : array_like, optional
        Output grid. Defaults to the source grid.
    log_interp : bool, optional
        Interpolate log10(PSD), which is usually better for spacecraft spectra.
    fill_value : float, optional
        Value outside the source grid.
    return_source_grid : bool, optional
        If True, also return the inverse-mapped source energy and pitch angle.

    Returns
    -------
    model_psd : ndarray
        Modeled consequence PSD on the target grid.
    """
    source, source_energy, source_pitch = _to_energy_pitch_array(source_psd, energy, pitch_angle)
    if target_energy is None:
        target_energy = source_energy
    if target_pitch_angle is None:
        target_pitch_angle = source_pitch

    target_energy = np.asarray(target_energy, dtype=float)
    target_pitch_angle = np.asarray(target_pitch_angle, dtype=float)
    e0, a0 = _inverse_energy_pitch(target_energy, target_pitch_angle, fermi_factor, betatron_factor)

    interp = _make_interpolator(source, source_energy, source_pitch, log_interp=log_interp, fill_value=fill_value)
    points = np.column_stack([e0.ravel(), a0.ravel()])
    modeled = interp(points).reshape(e0.shape)
    if log_interp:
        modeled = 10.0 ** modeled

    if return_source_grid:
        return modeled, e0, a0
    return modeled


def _fit_mask(energy, pitch_angle, target, model, energy_range, pitch_angle_range, log_space):
    ee, aa = np.meshgrid(energy, pitch_angle, indexing="ij")
    mask = np.isfinite(target) & np.isfinite(model)
    if log_space:
        mask &= (target > 0) & (model > 0)
    if energy_range is not None:
        mask &= (ee >= energy_range[0]) & (ee <= energy_range[1])
    if pitch_angle_range is not None:
        mask &= (aa >= pitch_angle_range[0]) & (aa <= pitch_angle_range[1])
    return mask


def _base_fit_mask(energy, pitch_angle, target, energy_range, pitch_angle_range, log_space):
    ee, aa = np.meshgrid(energy, pitch_angle, indexing="ij")
    mask = np.isfinite(target)
    if log_space:
        mask &= target > 0
    if energy_range is not None:
        mask &= (ee >= energy_range[0]) & (ee <= energy_range[1])
    if pitch_angle_range is not None:
        mask &= (aa >= pitch_angle_range[0]) & (aa <= pitch_angle_range[1])
    if not np.any(mask):
        raise ValueError("No valid target points remain for fitting.")
    return mask


def _metrics(target, model, mask, n_param, log_space):
    if log_space:
        y = np.log10(target[mask])
        y_fit = np.log10(model[mask])
    else:
        y = target[mask]
        y_fit = model[mask]

    residual = y_fit - y
    n = residual.size
    rss = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(rss / n)) if n else np.nan
    mae = float(np.mean(np.abs(residual))) if n else np.nan
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if n else np.nan
    r2 = float(1.0 - rss / ss_tot) if ss_tot and np.isfinite(ss_tot) else np.nan
    dof = n - n_param
    reduced_chi2 = float(rss / dof) if dof > 0 else np.nan
    return {
        "n_points": int(n),
        "rss": rss,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "reduced_chi2": reduced_chi2,
    }


def _error_value(target, model, mask, log_space=True, metric="rmse", normalize_energy=False):
    good = mask & np.isfinite(target) & np.isfinite(model)
    if log_space:
        good &= (target > 0) & (model > 0)
        target_work = np.full_like(target, np.nan, dtype=float)
        model_work = np.full_like(model, np.nan, dtype=float)
        target_work[good] = np.log10(target[good])
        model_work[good] = np.log10(model[good])
    else:
        target_work = np.full_like(target, np.nan, dtype=float)
        model_work = np.full_like(model, np.nan, dtype=float)
        target_work[good] = target[good]
        model_work[good] = model[good]

    if normalize_energy:
        target_med = _nanmedian_axis1(target_work)
        model_med = _nanmedian_axis1(model_work)
        target_work = target_work - target_med
        model_work = model_work - model_med
        good = mask & np.isfinite(target_work) & np.isfinite(model_work)

    y = target_work[good]
    y_fit = model_work[good]

    n_good = int(np.sum(good))
    if n_good == 0:
        return np.nan, n_good

    residual = y_fit - y
    if metric == "rmse":
        value = np.sqrt(np.mean(residual ** 2))
    elif metric == "mae":
        value = np.mean(np.abs(residual))
    elif metric == "rss":
        value = np.sum(residual ** 2)
    else:
        raise ValueError("metric must be 'rmse', 'mae', or 'rss'.")

    return float(value), n_good


def _nanmedian_axis1(arr):
    med = np.full((arr.shape[0], 1), np.nan, dtype=float)
    for i in range(arr.shape[0]):
        good = np.isfinite(arr[i, :])
        if np.any(good):
            med[i, 0] = np.nanmedian(arr[i, good])
    return med


def fermi_betatron_error_map(
    source_psd,
    target_psd,
    energy,
    pitch_angle,
    Fs=None,
    Fb=None,
    target_energy=None,
    target_pitch_angle=None,
    energy_range=None,
    pitch_angle_range=None,
    log_space=True,
    log_interp=True,
    metric="rmse",
    normalize_energy=False,
    min_valid_fraction=0.5,
    return_best_model=True,
):
    """Scan Fermi and betatron factors and return an error map.

    Parameters
    ----------
    source_psd, target_psd : array_like
        PSD arrays with shape (nenergy, npitchangle). They do not need to use
        the same energy grid.
    energy, pitch_angle : array_like
        Source energy in eV and source pitch angle in degrees.
    Fs, Fb : array_like, optional
        Fermi-factor and betatron-factor grids. Defaults to
        ``np.linspace(0.5, 2.0, 61)`` for both.
    target_energy, target_pitch_angle : array_like, optional
        Target grid. Defaults to ``energy`` and ``pitch_angle``.
    energy_range, pitch_angle_range : tuple, optional
        Target-grid range used to compute the error.
    log_space : bool, optional
        If True, compare log10(PSD). This is recommended for spectra.
    log_interp : bool, optional
        If True, interpolate log10(source PSD) during the mapping.
    metric : {"rmse", "mae", "rss"}, optional
        Error metric used for each factor pair.
    normalize_energy : bool, optional
        If True, subtract the pitch-angle median at each energy before
        computing the error. This focuses the scan on PAD shape instead of
        the absolute energy spectrum.
    min_valid_fraction : float, optional
        Minimum fraction of target-mask points that must have finite modeled
        values. Factor pairs below this fraction are assigned NaN.
    return_best_model : bool, optional
        If True, also return the model PSD for the minimum-error factor pair.

    Returns
    -------
    result : dict
        Dictionary containing ``error_map`` with shape ``(len(Fb), len(Fs))``,
        the factor grids, the best factors, valid-point counts, and optionally
        ``best_model_psd``.
    """
    source, source_energy, source_pitch = _to_energy_pitch_array(source_psd, energy, pitch_angle)

    if target_energy is None:
        target_energy = source_energy
    if target_pitch_angle is None:
        target_pitch_angle = source_pitch

    target, target_energy, target_pitch = _to_energy_pitch_array(
        target_psd,
        target_energy,
        target_pitch_angle,
    )

    if Fs is None:
        Fs = np.linspace(0.5, 2.0, 61)
    if Fb is None:
        Fb = np.linspace(0.5, 2.0, 61)
    Fs = np.asarray(Fs, dtype=float).ravel()
    Fb = np.asarray(Fb, dtype=float).ravel()
    if np.any(Fs <= 0) or np.any(Fb <= 0):
        raise ValueError("Fs and Fb must contain positive values.")

    base_mask = _base_fit_mask(
        target_energy,
        target_pitch,
        target,
        energy_range,
        pitch_angle_range,
        log_space,
    )
    n_base = int(np.sum(base_mask))
    min_valid_points = int(np.ceil(float(min_valid_fraction) * n_base))

    error_map = np.full((Fb.size, Fs.size), np.nan, dtype=float)
    n_valid_map = np.zeros((Fb.size, Fs.size), dtype=int)

    for j, fb in enumerate(Fb):
        for i, ff in enumerate(Fs):
            model = fermi_betatron_map(
                source,
                energy=source_energy,
                pitch_angle=source_pitch,
                fermi_factor=ff,
                betatron_factor=fb,
                target_energy=target_energy,
                target_pitch_angle=target_pitch,
                log_interp=log_interp,
            )
            value, n_good = _error_value(
                target,
                model,
                base_mask,
                log_space=log_space,
                metric=metric,
                normalize_energy=normalize_energy,
            )
            n_valid_map[j, i] = n_good
            if n_good >= min_valid_points:
                error_map[j, i] = value

    if np.all(~np.isfinite(error_map)):
        raise ValueError("No factor pair has enough valid modeled points.")

    best_j, best_i = np.unravel_index(np.nanargmin(error_map), error_map.shape)
    best_ff = float(Fs[best_i])
    best_fb = float(Fb[best_j])

    result = {
        "error_map": error_map,
        "n_valid_map": n_valid_map,
        "fermi_factors": Fs,
        "betatron_factors": Fb,
        "best_fermi_factor": best_ff,
        "best_betatron_factor": best_fb,
        "best_error": float(error_map[best_j, best_i]),
        "best_indices": (int(best_j), int(best_i)),
        "fit_mask": base_mask,
        "target_energy": target_energy,
        "target_pitch_angle": target_pitch,
        "metric": metric,
        "log_space": bool(log_space),
        "normalize_energy": bool(normalize_energy),
    }

    if return_best_model:
        result["best_model_psd"] = fermi_betatron_map(
            source,
            energy=source_energy,
            pitch_angle=source_pitch,
            fermi_factor=best_ff,
            betatron_factor=best_fb,
            target_energy=target_energy,
            target_pitch_angle=target_pitch,
            log_interp=log_interp,
        )

    return result


def fit_fermi_betatron(
    source_psd,
    target_psd,
    energy=None,
    pitch_angle=None,
    fit_fermi=True,
    fit_betatron=True,
    initial=(1.0, 1.0),
    bounds=((0.1, 0.1), (4.0, 4.0)),
    energy_range=None,
    pitch_angle_range=None,
    log_space=True,
    log_interp=True,
):
    """Fit Fermi and betatron factors from source and consequence PSDs.

    Parameters
    ----------
    source_psd, target_psd : array_like or xarray.DataArray
        Source and consequence PSDs on the same energy and pitch-angle grid.
    energy, pitch_angle : array_like
        Energy in eV and pitch angle in degrees.
    fit_fermi, fit_betatron : bool, optional
        Select which factors to fit. A fixed factor uses the corresponding
        value in ``initial``.
    initial : tuple, optional
        Initial (Ff, Fb).
    bounds : tuple, optional
        Lower and upper bounds as ((Ff_min, Fb_min), (Ff_max, Fb_max)).
    energy_range, pitch_angle_range : tuple, optional
        Restrict the fitting region.
    log_space : bool, optional
        Minimize residuals in log10(PSD). Recommended for spectra.
    log_interp : bool, optional
        Use log10 interpolation while mapping the source PSD.

    Returns
    -------
    result : dict
        Contains fitted factors, modeled PSD, residuals, metrics, and the
        scipy optimization result.
    """
    source, energy, pitch_angle = _to_energy_pitch_array(source_psd, energy, pitch_angle)
    target, target_energy, target_pitch = _to_energy_pitch_array(target_psd, energy, pitch_angle)
    if not (np.array_equal(energy, target_energy) and np.array_equal(pitch_angle, target_pitch)):
        raise ValueError("source_psd and target_psd must use the same energy and pitch-angle grid.")
    if not (fit_fermi or fit_betatron):
        raise ValueError("At least one of fit_fermi or fit_betatron must be True.")

    initial = np.asarray(initial, dtype=float)
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    active = np.array([fit_fermi, fit_betatron], dtype=bool)
    base_mask = _base_fit_mask(energy, pitch_angle, target, energy_range, pitch_angle_range, log_space)

    def unpack(x):
        factors = initial.copy()
        factors[active] = x
        return float(factors[0]), float(factors[1])

    def residual_fun(x):
        ff, fb = unpack(x)
        model = fermi_betatron_map(
            source,
            energy=energy,
            pitch_angle=pitch_angle,
            fermi_factor=ff,
            betatron_factor=fb,
            target_energy=energy,
            target_pitch_angle=pitch_angle,
            log_interp=log_interp,
        )
        if log_space:
            residual = np.full(int(np.sum(base_mask)), 30.0, dtype=float)
            good = base_mask & np.isfinite(model) & (model > 0)
            residual[np.ravel(good[base_mask])] = np.log10(model[good]) - np.log10(target[good])
            return residual

        residual = np.full(int(np.sum(base_mask)), np.nanmax(np.abs(target[base_mask])) * 10.0, dtype=float)
        good = base_mask & np.isfinite(model)
        residual[np.ravel(good[base_mask])] = model[good] - target[good]
        return residual

    opt = least_squares(
        residual_fun,
        x0=initial[active],
        bounds=(lower[active], upper[active]),
    )
    fermi_factor, betatron_factor = unpack(opt.x)
    model, e0, a0 = fermi_betatron_map(
        source,
        energy=energy,
        pitch_angle=pitch_angle,
        fermi_factor=fermi_factor,
        betatron_factor=betatron_factor,
        target_energy=energy,
        target_pitch_angle=pitch_angle,
        log_interp=log_interp,
        return_source_grid=True,
    )
    mask = _fit_mask(energy, pitch_angle, target, model, energy_range, pitch_angle_range, log_space)
    metrics = _metrics(target, model, mask, int(np.sum(active)), log_space)
    residual = np.full_like(target, np.nan, dtype=float)
    if log_space:
        residual[mask] = np.log10(model[mask]) - np.log10(target[mask])
    else:
        residual[mask] = model[mask] - target[mask]

    return {
        "fermi_factor": fermi_factor,
        "betatron_factor": betatron_factor,
        "model_psd": model,
        "residual": residual,
        "source_energy_grid": e0,
        "source_pitch_angle_grid": a0,
        "fit_mask": mask,
        "metrics": metrics,
        "optimization": opt,
    }


__all__ = [
    "def_to_psd_shape",
    "fermi_betatron_map",
    "fermi_betatron_error_map",
    "fit_fermi_betatron",
]

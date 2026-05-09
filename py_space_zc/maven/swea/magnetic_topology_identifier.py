#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infer magnetic topology from MAVEN SWEA pitch angle distributions.

This is a practical implementation of the Xu et al. (2019) combined idea:
identify photoelectrons in the two field-aligned directions and combine this
with loss-cone information from the pitch angle distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from py_space_zc import ts_scalar

from .photoelectron_identifier import (
    PhotoelectronConfig,
    _identify_one_spectrum,
)


TOPOLOGY_NAMES = {
    0: "unknown",
    1: "C-D",
    2: "C-X",
    3: "C-T",
    4: "C-V",
    5: "O-D",
    6: "O-N",
    7: "DP",
}


@dataclass(frozen=True)
class MagneticTopologyConfig:
    photoelectron: PhotoelectronConfig = PhotoelectronConfig()
    fa_pitch_range: tuple[float, float] = (0.0, 30.0)
    anti_fa_pitch_range: tuple[float, float] = (150.0, 180.0)
    perp_pitch_range: tuple[float, float] = (75.0, 105.0)
    loss_cone_energy_range_ev: tuple[float, float] = (100.0, 300.0)
    flux_ratio_energy_range_ev: tuple[float, float] = (35.0, 60.0)
    void_energy_ev: float = 40.0
    void_flux_threshold: float = 1.5e5
    loss_cone_ratio: float = 0.5
    closed_day_ratio_range: tuple[float, float] = (0.2, 5.0)


def magnetic_topology_identifier(
    pad,
    B=None,
    Pmso=None,
    swea_omni=None,
    config: MagneticTopologyConfig | None = None,
):
    """
    Infer magnetic topology from SWEA PAD data.

    Parameters
    ----------
    pad
        Dataset returned by ``maven.swea.get_pad(tint)``.
    B, Pmso
        Optional MAG field and spacecraft position in MSO. If provided, parallel
        and antiparallel populations are mapped to away and toward directions
        using the sign of ``B dot rhat``. If omitted, parallel is treated as
        away and antiparallel as toward.
    swea_omni
        Optional SWEA omni ts_spectr. If provided, this is used as the global
        photoelectron mask before directional PAD photoelectron flags are used.
    config
        Optional thresholds.

    Returns
    -------
    xarray.DataArray
        ``ts_scalar`` containing the magnetic topology index.
    """

    cfg = config or MagneticTopologyConfig()
    time, energy, pitchangle, data = _read_pad(pad)
    energy, data = _normalize_pad_energy(energy, data)

    par_idx = _pitch_indices(pitchangle, cfg.fa_pitch_range)
    anti_idx = _pitch_indices(pitchangle, cfg.anti_fa_pitch_range)
    perp_idx = _pitch_indices(pitchangle, cfg.perp_pitch_range)
    if par_idx.size == 0 or anti_idx.size == 0 or perp_idx.size == 0:
        raise ValueError("PAD pitch-angle bins do not cover field-aligned and perpendicular ranges")

    par_flux = np.nanmean(data[:, :, par_idx], axis=2)
    anti_flux = np.nanmean(data[:, :, anti_idx], axis=2)
    perp_flux = np.nanmean(data[:, :, perp_idx], axis=2)
    omni_flux = np.nanmean(data, axis=2)

    par_phe, par_feature = _direction_photoelectron_flags(energy, par_flux, cfg.photoelectron)
    anti_phe, anti_feature = _direction_photoelectron_flags(energy, anti_flux, cfg.photoelectron)
    if swea_omni is None:
        omni_phe, _ = _direction_photoelectron_flags(energy, omni_flux, cfg.photoelectron)
    else:
        omni_phe, _ = _omni_photoelectron_from_swea(time, swea_omni)
    par_phe = par_phe & omni_phe
    anti_phe = anti_phe & omni_phe

    par_lc, par_fa_perp_ratio = _loss_cone_flags(energy, par_flux, perp_flux, cfg)
    anti_lc, anti_fa_perp_ratio = _loss_cone_flags(energy, anti_flux, perp_flux, cfg)
    void_flag = _void_flags(energy, omni_flux, cfg)

    away_is_parallel = _away_parallel_flag(time, B, Pmso)
    sza = _sza_from_position(time, Pmso)
    nightside = np.isfinite(sza) & (sza > 90.0)
    away_phe = np.where(away_is_parallel, par_phe, anti_phe)
    toward_phe = np.where(away_is_parallel, anti_phe, par_phe)
    away_lc = np.where(away_is_parallel, par_lc, anti_lc)
    toward_lc = np.where(away_is_parallel, anti_lc, par_lc)
    away_flux = np.where(away_is_parallel[:, None], par_flux, anti_flux)
    toward_flux = np.where(away_is_parallel[:, None], anti_flux, par_flux)

    away_toward_ratio = _band_ratio_series(energy, away_flux, toward_flux, cfg.flux_ratio_energy_range_ev)
    topology_code = _classify_topology(
        away_phe,
        toward_phe,
        away_lc,
        toward_lc,
        void_flag,
        away_toward_ratio,
        nightside,
        cfg,
    )

    return ts_scalar(
        time,
        topology_code.astype(float),
        attrs={
            "name": "topology_index",
            "Instrument": "SWEA",
            "description": "Xu-style magnetic topology index",
            "index_map": TOPOLOGY_NAMES,
        },
    )


def _read_pad(pad):
    if not hasattr(pad, "data") or "time" not in pad:
        raise TypeError("pad must be the Dataset returned by maven.swea.get_pad")
    time = np.asarray(pad.time.data)
    energy = np.asarray(pad.energy.data, dtype=float)
    pitchangle = np.asarray(pad.pitchangle.data, dtype=float)
    data = np.asarray(pad.data.data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"PAD data must have shape (ntime, nE, nPA), got {data.shape}")
    return time, energy, pitchangle, data


def _normalize_pad_energy(energy, data):
    ntime, nE, nPA = data.shape
    if energy.ndim == 2:
        energy_use = energy
    elif energy.ndim == 1:
        energy_use = energy
    else:
        raise ValueError(f"energy must be 1D or 2D, got {energy.shape}")

    if energy_use.ndim == 1:
        energy_ev = _as_ev(energy_use)
        order = np.argsort(energy_ev)
        return energy_ev[order], data[:, order, :]

    energy_sorted = np.empty_like(energy_use)
    data_sorted = np.empty_like(data)
    for i in range(ntime):
        order = np.argsort(_as_ev(energy_use[i, :]))
        energy_sorted[i, :] = _as_ev(energy_use[i, :])[order]
        data_sorted[i, :, :] = data[i, order, :]
    return energy_sorted, data_sorted


def _pitch_indices(pitchangle, pitch_range):
    return np.where((pitchangle >= pitch_range[0]) & (pitchangle <= pitch_range[1]))[0]


def _direction_photoelectron_flags(energy, flux, photo_cfg):
    ntime = flux.shape[0]
    flag = np.zeros(ntime, dtype=bool)
    feature = np.zeros(ntime)
    for i in range(ntime):
        Ei = energy[i, :] if energy.ndim == 2 else energy
        result = _identify_one_spectrum(Ei, flux[i, :], photo_cfg)
        feature[i] = result["pp_flag"] + result["knee_flag"] + result["auger_flag"]
        flag[i] = (result["pp_flag"] == 1) and (feature[i] >= photo_cfg.min_feature_count)
    return flag, feature


def _omni_photoelectron_from_swea(time, swea_omni):
    from .photoelectron_identifier import photoelectron_identifier

    omni = photoelectron_identifier(swea_omni)
    t_src = np.asarray(omni["is_photoelectron"].time.data).astype("datetime64[ns]").astype(np.int64)
    t_dst = np.asarray(time).astype("datetime64[ns]").astype(np.int64)
    phe = np.asarray(omni["is_photoelectron"].data, dtype=float)
    nearest = np.searchsorted(t_src, t_dst)
    nearest = np.clip(nearest, 1, t_src.size - 1)
    left = nearest - 1
    choose_left = np.abs(t_dst - t_src[left]) <= np.abs(t_dst - t_src[nearest])
    idx = np.where(choose_left, left, nearest)
    return phe[idx] == 1, np.full(t_dst.size, np.nan)


def _loss_cone_flags(energy, fa_flux, perp_flux, cfg):
    fa = _band_mean_series(energy, fa_flux, cfg.loss_cone_energy_range_ev)
    perp = _band_mean_series(energy, perp_flux, cfg.loss_cone_energy_range_ev)
    ratio = fa / perp
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio < cfg.loss_cone_ratio, ratio


def _void_flags(energy, omni_flux, cfg):
    flux_40 = _interp_energy_series(energy, omni_flux, cfg.void_energy_ev)
    return np.isfinite(flux_40) & (flux_40 < cfg.void_flux_threshold)


def _band_ratio_series(energy, num_flux, den_flux, band):
    num = _band_mean_series(energy, num_flux, band)
    den = _band_mean_series(energy, den_flux, band)
    ratio = num / den
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def _band_mean_series(energy, flux, band):
    out = np.full(flux.shape[0], np.nan)
    for i in range(flux.shape[0]):
        Ei = energy[i, :] if energy.ndim == 2 else energy
        mask = (Ei >= band[0]) & (Ei <= band[1])
        if np.any(mask):
            out[i] = np.nanmean(flux[i, mask])
    return out


def _interp_energy_series(energy, flux, target_energy):
    out = np.full(flux.shape[0], np.nan)
    for i in range(flux.shape[0]):
        Ei = energy[i, :] if energy.ndim == 2 else energy
        good = np.isfinite(Ei) & np.isfinite(flux[i, :]) & (Ei > 0) & (flux[i, :] > 0)
        if np.count_nonzero(good) >= 2:
            out[i] = np.exp(np.interp(np.log(target_energy), np.log(Ei[good]), np.log(flux[i, good])))
    return out


def _away_parallel_flag(time, B, Pmso):
    if B is None or Pmso is None:
        return np.ones(time.size, dtype=bool)
    B_arr = _interp_vector_to_time(time, B)
    P_arr = _interp_vector_to_time(time, Pmso)
    rnorm = np.linalg.norm(P_arr, axis=1)
    rhat = P_arr / rnorm[:, None]
    br = np.sum(B_arr * rhat, axis=1)
    return br >= 0


def _sza_from_position(time, Pmso):
    if Pmso is None:
        return np.full(time.size, np.nan)
    P_arr = _interp_vector_to_time(time, Pmso)
    rnorm = np.linalg.norm(P_arr, axis=1)
    return np.degrees(np.arccos(np.clip(P_arr[:, 0] / rnorm, -1.0, 1.0)))


def _interp_vector_to_time(time, vec):
    if isinstance(vec, dict):
        if "Bmso" in vec:
            vec = vec["Bmso"]
        elif "Pmso" in vec:
            vec = vec["Pmso"]
    if not hasattr(vec, "time"):
        raise TypeError("B and Pmso must be xarray DataArray objects or a MAG dict")
    t_src = np.asarray(vec.time.data).astype("datetime64[ns]").astype(np.int64)
    t_dst = np.asarray(time).astype("datetime64[ns]").astype(np.int64)
    arr = np.asarray(vec.data, dtype=float)
    return np.vstack([np.interp(t_dst, t_src, arr[:, j]) for j in range(arr.shape[1])]).T


def _classify_topology(away_phe, toward_phe, away_lc, toward_lc, void, ratio, nightside, cfg):
    code = np.zeros(away_phe.size, dtype=int)
    low, high = cfg.closed_day_ratio_range

    # Voids are identified directly by low suprathermal electron flux.
    code[void] = 4
    un = code == 0
    code[un & away_lc & toward_lc] = 3

    un = code == 0
    both_phe = away_phe & toward_phe
    cd = both_phe & np.isfinite(ratio) & (ratio >= low) & (ratio <= high)
    code[un & cd] = 1

    un = code == 0
    cx = both_phe & (~cd)
    cx |= away_phe & (~toward_phe) & toward_lc
    cx |= (~away_phe) & toward_phe & away_lc
    code[un & cx] = 2

    un = code == 0
    od = away_phe & (~toward_phe) & (~toward_lc)
    od |= (~away_phe) & toward_phe & (~away_lc)
    code[un & od] = 5

    un = code == 0
    on = (~away_phe) & (~toward_phe) & away_lc & (~toward_lc) & nightside
    code[un & on] = 6

    un = code == 0
    dp = (~away_phe) & (~toward_phe) & (~away_lc)
    code[un & dp] = 7
    return code


def _as_ev(energy):
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        return energy
    return energy * 1000.0 if np.nanmax(finite) <= 10.0 else energy


__all__ = ["MagneticTopologyConfig", "magnetic_topology_identifier", "TOPOLOGY_NAMES"]

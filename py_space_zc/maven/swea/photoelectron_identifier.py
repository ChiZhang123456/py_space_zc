#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Identify Martian ionospheric photoelectrons from MAVEN SWEA spectra.

Main usage
----------
>>> from py_space_zc import maven
>>> tint = ["2018-11-01T01:00:00", "2018-11-01T02:00:00"]
>>> swea = maven.get_data(tint, "swea_omni")
>>> res = maven.swea.photoelectron_identifier(swea)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from py_space_zc import ts_scalar


@dataclass(frozen=True)
class PhotoelectronConfig:
    heii_range_ev: tuple[float, float] = (19.0, 30.0)
    knee_low_ev: tuple[float, float] = (45.0, 60.0)
    knee_high_ev: tuple[float, float] = (70.0, 90.0)
    auger_c_range_ev: tuple[float, float] = (220.0, 290.0)
    auger_o_range_ev: tuple[float, float] = (430.0, 560.0)
    min_heii_prominence: float = 0.14
    min_knee_drop: float = 2.0
    min_auger_prominence: float = 0.50
    min_feature_count: int = 2
    continuity_window: int = 10
    continuity_min_count: int = 5


def photoelectron_identifier(
    swea,
    config: PhotoelectronConfig | None = None,
    require_continuity: bool = True,
):
    """
    Identify photoelectrons from a SWEA ts_spectr object.

    Parameters
    ----------
    swea
        SWEA spectrum, normally returned by ``maven.get_data(tint, "swea_omni")``.
        The data must be shaped as ``(ntime, nE)``. The energy coordinate can be
        ``(nE,)``, ``(nE, 1)``, ``(1, nE)``, or ``(ntime, nE)``.
    config
        Optional thresholds.
    require_continuity
        If True, ``is_photoelectron`` also requires several nearby positive
        spectra, following Wang et al. (2022)'s PEB crossing continuity idea.

    Returns
    -------
    dict
        Dictionary of ts_scalar objects. Flags are 0 or 1.
    """

    cfg = config or PhotoelectronConfig()
    time, energy, DEF = _read_swea_ts_spectr(swea)
    energy, DEF = _normalize_energy_def(energy, DEF)

    out = {
        "pp_flag": np.zeros(DEF.shape[0]),
        "knee_flag": np.zeros(DEF.shape[0]),
        "auger_flag": np.zeros(DEF.shape[0]),
        "heii_peak_ev": np.full(DEF.shape[0], np.nan),
        "heii_prominence": np.full(DEF.shape[0], np.nan),
        "knee_drop_ratio": np.full(DEF.shape[0], np.nan),
        "auger_c_prominence": np.full(DEF.shape[0], np.nan),
        "auger_o_prominence": np.full(DEF.shape[0], np.nan),
    }

    for i in range(DEF.shape[0]):
        Ei = energy[i, :] if energy.ndim == 2 else energy
        result = _identify_one_spectrum(Ei, DEF[i, :], cfg)
        for key, value in result.items():
            out[key][i] = value

    feature_count = out["pp_flag"] + out["knee_flag"] + out["auger_flag"]
    phe = (out["pp_flag"] == 1) & (feature_count >= cfg.min_feature_count)
    if require_continuity:
        phe = phe & _continuity_flag(phe, cfg.continuity_window, cfg.continuity_min_count)
    out["is_photoelectron"] = phe.astype(float)
    public_keys = ["is_photoelectron", "pp_flag", "knee_flag", "auger_flag"]

    return {
        key: ts_scalar(
            time,
            out[key].astype(float),
            attrs={
                "name": key,
                "Instrument": "SWEA",
                "description": "Photoelectron diagnostic from SWEA spectra",
            },
        )
        for key in public_keys
    }


def plot_photoelectron_spectrum(
    swea,
    index: int | None = None,
    target_time=None,
    ax=None,
    config: PhotoelectronConfig | None = None,
):
    """
    Plot one SWEA spectrum and annotate photoelectron feature diagnostics.

    Parameters
    ----------
    swea
        SWEA ts_spectr object.
    index
        Time index to plot. If omitted, ``target_time`` is used. If both are
        omitted, the first spectrum classified as photoelectron is plotted.
    target_time
        Time nearest to the requested value. Accepts ``numpy.datetime64`` or a
        string readable by ``numpy.datetime64``.
    ax
        Optional matplotlib axis.
    config
        Optional thresholds.

    Returns
    -------
    fig, ax, info
        ``info`` contains the selected time, index, and diagnostic values.
    """

    import matplotlib.pyplot as plt

    cfg = config or PhotoelectronConfig()
    time, energy, DEF = _read_swea_ts_spectr(swea)
    energy, DEF = _normalize_energy_def(energy, DEF)

    if index is None:
        if target_time is not None:
            target = np.datetime64(target_time)
            index = int(np.argmin(np.abs(time - target)))
        else:
            res = photoelectron_identifier(swea, config=cfg)
            phe = np.asarray(res["is_photoelectron"].data) == 1
            index = int(np.argmax(phe)) if np.any(phe) else 0

    Ei = energy[index, :] if energy.ndim == 2 else energy
    Fi = DEF[index, :]
    result = _identify_one_spectrum(Ei, Fi, cfg)
    feature_count = result["pp_flag"] + result["knee_flag"] + result["auger_flag"]
    result["feature_count"] = feature_count
    result["is_photoelectron"] = float(
        (result["pp_flag"] == 1) and (feature_count >= cfg.min_feature_count)
    )
    result["time"] = time[index]
    result["index"] = index

    good = np.isfinite(Ei) & np.isfinite(Fi) & (Ei > 0) & (Fi > 0)
    Ei = Ei[good]
    Fi = Fi[good]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    else:
        fig = ax.figure

    ax.loglog(Ei, Fi, color="black", marker="o", ms=3.0, lw=1.1, drawstyle="steps-mid")
    _shade_band(ax, cfg.heii_range_ev, "He II peak\n19-30 eV", "tab:blue")
    _shade_band(ax, (60.0, 70.0), "knee\n60-70 eV", "tab:green")
    _shade_band(ax, cfg.auger_c_range_ev, "C Auger\n~250 eV", "tab:orange")
    _shade_band(ax, cfg.auger_o_range_ev, "O Auger\n~500 eV", "tab:red")

    text = (
        f"time = {np.datetime_as_string(time[index], unit='s')}\n"
        f"PP = {int(result['pp_flag'])}, He II prominence = {result['heii_prominence']:.2f}\n"
        f"knee = {int(result['knee_flag'])}, knee ratio = {result['knee_drop_ratio']:.2f}\n"
        f"Auger = {int(result['auger_flag'])}, C/O prom = "
        f"{result['auger_c_prominence']:.2f}/{result['auger_o_prominence']:.2f}\n"
        f"feature count = {int(feature_count)}, PHE = {int(result['is_photoelectron'])}"
    )
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86, "edgecolor": "0.75"},
    )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("SWEA DEF")
    ax.set_title("SWEA photoelectron feature check")
    ax.grid(True, which="both", alpha=0.25)
    return fig, ax, result


def _shade_band(ax, band, label, color):
    ax.axvspan(band[0], band[1], color=color, alpha=0.14)
    x = np.sqrt(band[0] * band[1])
    ax.text(
        x,
        0.97,
        label,
        color=color,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
    )


def _read_swea_ts_spectr(swea):
    if not hasattr(swea, "coords") or not hasattr(swea, "data"):
        raise TypeError("swea must be a ts_spectr-like xarray.DataArray")
    if "time" not in swea.coords:
        raise ValueError("swea must have a time coordinate")

    energy_name = "energy" if "energy" in swea.coords else swea.dims[-1]
    time = np.asarray(swea.time.data)
    energy = np.asarray(swea.coords[energy_name].data, dtype=float)
    DEF = np.asarray(swea.data, dtype=float)
    return time, energy, DEF


def _normalize_energy_def(energy, DEF):
    if DEF.ndim != 2:
        raise ValueError(f"DEF must have shape (ntime, nE), got {DEF.shape}")

    ntime, nE = DEF.shape
    energy = np.asarray(energy, dtype=float)

    if energy.ndim == 1:
        energy_use = energy
    elif energy.ndim == 2 and 1 in energy.shape:
        energy_use = energy.reshape(-1)
    elif energy.ndim == 2 and energy.shape == DEF.shape:
        energy_use = energy
    else:
        raise ValueError(
            "energy must have shape (nE,), (nE, 1), (1, nE), or (ntime, nE); "
            f"got {energy.shape}"
        )

    if energy_use.ndim == 1:
        if energy_use.size != nE:
            raise ValueError(f"energy length {energy_use.size} does not match DEF nE {nE}")
        energy_ev = _as_ev(energy_use)
        order = np.argsort(energy_ev)
        return energy_ev[order], DEF[:, order]

    if energy_use.shape != DEF.shape:
        raise ValueError(f"2D energy shape {energy_use.shape} must match DEF shape {DEF.shape}")

    energy_ev = _as_ev(energy_use)
    energy_sorted = np.empty_like(energy_ev)
    DEF_sorted = np.empty_like(DEF)
    for i in range(ntime):
        order = np.argsort(energy_ev[i, :])
        energy_sorted[i, :] = energy_ev[i, order]
        DEF_sorted[i, :] = DEF[i, order]
    return energy_sorted, DEF_sorted


def _identify_one_spectrum(energy, flux, cfg):
    good = np.isfinite(energy) & np.isfinite(flux) & (energy > 0) & (flux > 0)
    energy = energy[good]
    flux = flux[good]
    if energy.size < 6:
        return _empty_result()

    pp_flag, heii_peak_ev, heii_prominence = _heii_peak_flag(energy, flux, cfg)

    knee_drop_ratio = _median_ratio(energy, flux, cfg.knee_low_ev, cfg.knee_high_ev)
    knee_flag = np.isfinite(knee_drop_ratio) and knee_drop_ratio >= cfg.min_knee_drop

    auger_c = _peak_prominence(energy, flux, cfg.auger_c_range_ev)
    auger_o = _peak_prominence(energy, flux, cfg.auger_o_range_ev)
    auger_flag = np.nanmax([auger_c, auger_o]) >= cfg.min_auger_prominence

    return {
        "pp_flag": float(pp_flag),
        "knee_flag": float(knee_flag),
        "auger_flag": float(auger_flag),
        "heii_peak_ev": heii_peak_ev,
        "heii_prominence": heii_prominence,
        "knee_drop_ratio": knee_drop_ratio,
        "auger_c_prominence": auger_c,
        "auger_o_prominence": auger_o,
    }


def _heii_peak_flag(energy, flux, cfg):
    idx = np.where((energy >= cfg.heii_range_ev[0]) & (energy <= cfg.heii_range_ev[1]))[0]
    idx = idx[(idx > 0) & (idx < energy.size - 1)]
    if idx.size == 0:
        return False, np.nan, np.nan

    peak_ev = np.nan
    best_prominence = np.nan
    has_local_peak = False

    for j in idx:
        background = 0.5 * (flux[j - 1] + flux[j + 1])
        prominence = (flux[j] - background) / background if background > 0 else np.nan
        if flux[j] > flux[j - 1] and flux[j] > flux[j + 1]:
            has_local_peak = True
        if not np.isfinite(best_prominence) or prominence > best_prominence:
            best_prominence = prominence
            peak_ev = energy[j]

    has_prominent_peak = (
        np.isfinite(best_prominence) and best_prominence >= cfg.min_heii_prominence
    )
    return bool(has_local_peak and has_prominent_peak), peak_ev, best_prominence


def _median_ratio(energy, flux, low_band, high_band):
    low = _median_band(energy, flux, low_band)
    high = _median_band(energy, flux, high_band)
    return low / high if np.isfinite(low) and np.isfinite(high) and high > 0 else np.nan


def _median_band(energy, flux, band):
    mask = (energy >= band[0]) & (energy <= band[1])
    return float(np.nanmedian(flux[mask])) if np.any(mask) else np.nan


def _peak_prominence(energy, flux, band):
    idx = np.where((energy >= band[0]) & (energy <= band[1]))[0]
    if idx.size == 0:
        return np.nan

    j = idx[np.nanargmax(flux[idx])]
    left = np.arange(max(0, idx[0] - 2), idx[0])
    right = np.arange(idx[-1] + 1, min(energy.size, idx[-1] + 3))
    background_idx = np.r_[left, right]
    if background_idx.size == 0:
        return np.nan

    background = np.nanmedian(flux[background_idx])
    return float((flux[j] - background) / background) if background > 0 else np.nan


def _continuity_flag(flag, window, min_count):
    out = np.zeros(flag.size, dtype=bool)
    for i in range(flag.size):
        out[i] = np.count_nonzero(flag[i : i + window]) >= min_count
    return out


def _as_ev(energy):
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        return energy
    return energy * 1000.0 if np.nanmax(finite) <= 10.0 else energy


def _empty_result():
    return {
        "pp_flag": 0.0,
        "knee_flag": 0.0,
        "auger_flag": 0.0,
        "heii_peak_ev": np.nan,
        "heii_prominence": np.nan,
        "knee_drop_ratio": np.nan,
        "auger_c_prominence": np.nan,
        "auger_o_prominence": np.nan,
    }


__all__ = ["PhotoelectronConfig", "photoelectron_identifier", "plot_photoelectron_spectrum"]

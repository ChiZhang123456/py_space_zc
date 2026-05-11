"""PCW detection with MAVEN magnetic-field SVD products.

This module keeps only the current H+ PCW workflow:

1. run ``SVD_B`` once on the full input ``Bwave`` interval,
2. compute local ``B0`` windows with ``background_B`` using the same
   window length and overlap,
3. evaluate the PSD, theta_kB, ellipticity, and duration criteria for each
   local B0 window,
4. optionally plot the diagnostic products.
"""

from __future__ import annotations

from typing import Any

import numpy as np


E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
ION_TABLE = {
    "H+": (1.007276, 1.0),
    "H": (1.007276, 1.0),
    "He+": (4.002602, 1.0),
    "He": (4.002602, 1.0),
    "O+": (15.999, 1.0),
    "O": (15.999, 1.0),
    "O2+": (31.998, 1.0),
    "O2": (31.998, 1.0),
    "CO2+": (44.0095, 1.0),
    "CO2": (44.0095, 1.0),
}


def _ion_mass_charge(ion: str | tuple[float, float]) -> tuple[float, float]:
    if isinstance(ion, tuple):
        mass_amu, charge_state = ion
    else:
        key = ion.strip()
        if key not in ION_TABLE:
            raise ValueError(f"unknown ion {ion!r}; use one of {sorted(ION_TABLE)} or pass (mass_amu, charge_state)")
        mass_amu, charge_state = ION_TABLE[key]
    return float(mass_amu) * AMU, float(charge_state) * E_CHARGE


def _time64_to_datetime64(text: str) -> np.datetime64:
    return np.datetime64(text.replace("/", "T"))


def _time_text(value: Any) -> str:
    return np.datetime_as_string(np.asarray(value).astype("datetime64[s]"), unit="s")


def _extract_bwave_arrays(bwave: Any) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(bwave, "time") or not hasattr(bwave, "data"):
        raise TypeError("Bwave must be an xarray DataArray with time and 3-component data")
    time = np.asarray(bwave.time.data)
    data = np.asarray(bwave.data, dtype=float)
    if data.ndim != 2 or 3 not in data.shape:
        raise ValueError("Bwave data must be a 2D array with one dimension of length 3")
    if data.shape[0] == 3 and data.shape[1] != 3:
        data = data.T
    if data.shape[1] != 3:
        raise ValueError("cannot identify the 3-component dimension in Bwave")
    if len(time) != data.shape[0]:
        raise ValueError("Bwave time coordinate length does not match data length")
    return time, data


def read_sw_interval_list(path: str, min_duration_s: float = 40.0 * 60.0) -> list[dict[str, Any]]:
    """Read the CNN solar-wind interval list."""
    intervals: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                index = int(parts[0])
                start = _time64_to_datetime64(parts[1])
                stop = _time64_to_datetime64(parts[2])
            except (TypeError, ValueError):
                continue
            duration_s = float((stop - start) / np.timedelta64(1, "s"))
            if duration_s >= min_duration_s:
                intervals.append(
                    {
                        "index": index,
                        "start": start,
                        "stop": stop,
                        "duration_s": duration_s,
                        "line_no": line_no,
                    }
                )
    return intervals


def _band_nanmean(values: np.ndarray, freq: np.ndarray, f_low: float, f_high: float) -> float:
    band = (freq >= f_low) & (freq <= f_high)
    if not np.any(band):
        return np.nan
    data = np.asarray(values)[..., band]
    if not np.any(np.isfinite(data)):
        return np.nan
    return float(np.nanmean(data))


def _time_band_nanmean(values: np.ndarray, freq: np.ndarray, f_low: float, f_high: float) -> np.ndarray:
    band = (freq >= f_low) & (freq <= f_high)
    out = np.full(np.asarray(values).shape[0], np.nan, dtype=float)
    if not np.any(band):
        return out
    data = np.asarray(values, dtype=float)[:, band]
    good = np.any(np.isfinite(data), axis=1)
    out[good] = np.nanmean(data[good], axis=1)
    return out


def _longest_true_duration_s(mask: np.ndarray, time: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0
    time = np.asarray(time).astype("datetime64[ns]")
    if time.size < 2:
        return 0.0
    dt_s = np.nanmedian(np.diff(time).astype("timedelta64[ns]").astype(float)) * 1.0e-9
    if not np.isfinite(dt_s) or dt_s <= 0:
        return 0.0
    best = 0
    current = 0
    for item in mask:
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best * dt_s)


def evaluate_pcw_svd_psd_criteria_window(
    wave_res: dict[str, Any],
    start: Any,
    stop: Any,
    b0_vec_nt: np.ndarray,
    ion: str | tuple[float, float] = "H+",
    *,
    min_duration_gyroperiods: float = 3.0,
    sideband_ratio_min: float = 1.2,
    perp_para_ratio_min: float = 2.0,
    theta_kb_max_deg: float = 45.0,
    ellipticity_max: float = -0.6,
) -> dict[str, Any]:
    """Evaluate the PCW criteria for one local B0 window using precomputed SVD_B output."""
    start = np.datetime64(start)
    stop = np.datetime64(stop)
    b0_vec = np.asarray(b0_vec_nt, dtype=float)
    b0_abs_nt = float(np.linalg.norm(b0_vec))
    if b0_vec.size != 3 or not np.isfinite(b0_abs_nt) or b0_abs_nt <= 0:
        return _empty_window_result("bad_b0")

    mass, charge = _ion_mass_charge(ion)
    fci_hz = abs(charge) * b0_abs_nt * 1e-9 / (2.0 * np.pi * mass)
    tci_s = 1.0 / fci_hz if fci_hz > 0 else np.nan

    svd_time = np.asarray(wave_res["Bperp"].time.data)
    smask = (svd_time >= start) & (svd_time < stop)
    freq = np.asarray(wave_res["Bperp"].coords["frequency"].data, dtype=float)
    bperp = np.asarray(wave_res["Bperp"].values, dtype=float)[smask]
    bpara = np.asarray(wave_res["Bpara"].values, dtype=float)[smask]
    theta = np.asarray(wave_res["theta"].values, dtype=float)[smask]
    ellipticity = np.asarray(wave_res["ellipticity"].values, dtype=float)[smask]
    local_time = svd_time[smask]

    low1, high1 = 0.6 * fci_hz, 0.8 * fci_hz
    low_main, high_main = 0.8 * fci_hz, 1.2 * fci_hz
    low2, high2 = 1.2 * fci_hz, 1.4 * fci_hz

    psd_bperp_low = _band_nanmean(bperp, freq, low1, high1)
    psd_bperp_main = _band_nanmean(bperp, freq, low_main, high_main)
    psd_bperp_high = _band_nanmean(bperp, freq, low2, high2)
    psd_bpara_main = _band_nanmean(bpara, freq, low_main, high_main)
    theta_main = _band_nanmean(theta, freq, low_main, high_main)
    ellipticity_main = _band_nanmean(ellipticity, freq, low_main, high_main)

    bperp_low_t = _time_band_nanmean(bperp, freq, low1, high1)
    bperp_main_t = _time_band_nanmean(bperp, freq, low_main, high_main)
    bperp_high_t = _time_band_nanmean(bperp, freq, low2, high2)
    bpara_main_t = _time_band_nanmean(bpara, freq, low_main, high_main)
    theta_t = _time_band_nanmean(theta, freq, low_main, high_main)
    ell_t = _time_band_nanmean(ellipticity, freq, low_main, high_main)

    with np.errstate(divide="ignore", invalid="ignore"):
        low_ratio = psd_bperp_main / psd_bperp_low
        high_ratio = psd_bperp_main / psd_bperp_high
        para_ratio = psd_bperp_main / psd_bpara_main
        low_ratio_t = bperp_main_t / bperp_low_t
        high_ratio_t = bperp_main_t / bperp_high_t
        para_ratio_t = bperp_main_t / bpara_main_t

    good_t = (
        np.isfinite(low_ratio_t)
        & np.isfinite(high_ratio_t)
        & np.isfinite(para_ratio_t)
        & (low_ratio_t >= sideband_ratio_min)
        & (high_ratio_t >= sideband_ratio_min)
        & (para_ratio_t >= perp_para_ratio_min)
        & (theta_t <= theta_kb_max_deg)
        & (ell_t < ellipticity_max)
    )
    continuous_duration_s = _longest_true_duration_s(good_t, local_time)
    required_duration_s = float(min_duration_gyroperiods * tci_s) if np.isfinite(tci_s) else np.inf

    psd_ok = bool(
        np.isfinite(low_ratio)
        and np.isfinite(high_ratio)
        and np.isfinite(para_ratio)
        and low_ratio >= sideband_ratio_min
        and high_ratio >= sideband_ratio_min
        and para_ratio >= perp_para_ratio_min
    )
    theta_ok = bool(np.isfinite(theta_main) and theta_main <= theta_kb_max_deg)
    ellipticity_ok = bool(np.isfinite(ellipticity_main) and ellipticity_main < ellipticity_max)
    duration_ok = bool(continuous_duration_s >= required_duration_s)
    is_icw = bool(psd_ok and theta_ok and ellipticity_ok and duration_ok)

    return {
        "is_icw": is_icw,
        "reason": "pass" if is_icw else "criteria_not_met",
        "B0x_mso_nT": float(b0_vec[0]),
        "B0y_mso_nT": float(b0_vec[1]),
        "B0z_mso_nT": float(b0_vec[2]),
        "B0_abs_nT": b0_abs_nt,
        "fci_hz": float(fci_hz),
        "Tci_s": float(tci_s),
        "band_low_hz": float(low_main),
        "band_high_hz": float(high_main),
        "theta_kb_deg": float(theta_main) if np.isfinite(theta_main) else np.nan,
        "ellipticity": float(ellipticity_main) if np.isfinite(ellipticity_main) else np.nan,
        "psd_bperp_0p6_0p8_fci": float(psd_bperp_low) if np.isfinite(psd_bperp_low) else np.nan,
        "psd_bperp_0p8_1p2_fci": float(psd_bperp_main) if np.isfinite(psd_bperp_main) else np.nan,
        "psd_bperp_1p2_1p4_fci": float(psd_bperp_high) if np.isfinite(psd_bperp_high) else np.nan,
        "psd_bpara_0p8_1p2_fci": float(psd_bpara_main) if np.isfinite(psd_bpara_main) else np.nan,
        "perp_main_over_low_ratio": float(low_ratio) if np.isfinite(low_ratio) else np.nan,
        "perp_main_over_high_ratio": float(high_ratio) if np.isfinite(high_ratio) else np.nan,
        "perp_over_para_main_ratio": float(para_ratio) if np.isfinite(para_ratio) else np.nan,
        "continuous_duration_s": continuous_duration_s,
        "required_duration_s": required_duration_s,
        "good_svd_samples": int(np.count_nonzero(good_t)),
        "svd_sample_count": int(np.count_nonzero(smask)),
        "psd_ok": psd_ok,
        "theta_ok": theta_ok,
        "ellipticity_ok": ellipticity_ok,
        "duration_ok": duration_ok,
    }


def _empty_window_result(reason: str) -> dict[str, Any]:
    return {
        "is_icw": False,
        "reason": reason,
        "theta_kb_deg": np.nan,
        "ellipticity": np.nan,
        "psd_bperp_0p8_1p2_fci": np.nan,
        "psd_bpara_0p8_1p2_fci": np.nan,
    }


def detect_pcw_svd_psd_criteria(
    Bwave: Any,
    ion: str | tuple[float, float] = "H+",
    *,
    tint_focus: tuple[str, str] | list[str] | tuple[np.datetime64, np.datetime64] | None = None,
    svd_window_length: float = 300.0,
    svd_overlap: float = 150.0,
    freq_range: tuple[float, float] | list[float] | None = None,
    m_width_coeff: float = 1,
    nav: int = 12,
    min_duration_gyroperiods: float = 3.0,
    sideband_ratio_min: float = 1.2,
    perp_para_ratio_min: float = 2.0,
    theta_kb_max_deg: float = 45.0,
    ellipticity_max: float = -0.6,
    include_partial_background: bool = False,
    plot: bool = False,
    plot_freq_range: tuple[float, float] = (0.01, 11.0),
) -> dict[str, Any]:
    """Run SVD_B on the full Bwave interval and return PCW criteria time series."""
    time, b_xyz_nt = _extract_bwave_arrays(Bwave)
    if time.size < 2:
        return {"is_icw": np.asarray([], dtype=bool), "reason": "too_few_samples"}

    sample_dt_s = np.nanmedian(np.diff(time.astype("datetime64[ns]")).astype("timedelta64[ns]").astype(float)) * 1.0e-9
    sample_dt_s = float(sample_dt_s) if np.isfinite(sample_dt_s) and sample_dt_s > 0 else 0.0
    duration_s = float((time[-1] - time[0]) / np.timedelta64(1, "s")) + sample_dt_s
    if duration_s < svd_window_length:
        return {
            "is_icw": np.asarray([], dtype=bool),
            "reason": "shorter_than_svd_window_length",
            "duration_s": duration_s,
        }

    mass, charge = _ion_mass_charge(ion)
    b0_global = float(np.linalg.norm(np.nanmean(b_xyz_nt, axis=0)))
    if not np.isfinite(b0_global) or b0_global <= 0:
        return {"is_icw": np.asarray([], dtype=bool), "reason": "bad_b0"}
    fci_global = abs(charge) * b0_global * 1e-9 / (2.0 * np.pi * mass)

    from py_space_zc.background_B import background_B
    from py_space_zc.method.SVD_B import SVD_B
    from py_space_zc.ts_scalar import ts_scalar
    from py_space_zc.ts_vec_xyz import ts_vec_xyz

    if freq_range is None:
        freq_range = [max(0.5 * fci_global, 1.0e-4), max(1.5 * fci_global, 1.2e-4)]

    clean_bwave = Bwave.dropna(dim="time") if hasattr(Bwave, "dropna") else Bwave
    wave_res = SVD_B(
        clean_bwave,
        window_length=svd_window_length,
        overlap=svd_overlap,
        freq_range=list(freq_range),
        m_width_coeff=m_width_coeff,
        nav=nav,
    )
    B0_all = background_B(
        clean_bwave,
        window_length=svd_window_length,
        overlap=svd_overlap,
        include_partial=include_partial_background,
    )

    b0_time = np.asarray(B0_all.time.data)
    half_width = np.timedelta64(int(round(0.5 * svd_window_length)), "s")
    rows: list[dict[str, Any]] = []
    for center, b0_vec in zip(b0_time, np.asarray(B0_all.data, dtype=float)):
        row = evaluate_pcw_svd_psd_criteria_window(
            wave_res,
            np.datetime64(center) - half_width,
            np.datetime64(center) + half_width,
            b0_vec,
            ion=ion,
            min_duration_gyroperiods=min_duration_gyroperiods,
            sideband_ratio_min=sideband_ratio_min,
            perp_para_ratio_min=perp_para_ratio_min,
            theta_kb_max_deg=theta_kb_max_deg,
            ellipticity_max=ellipticity_max,
        )
        row["window_center"] = np.datetime64(center).astype("datetime64[ns]")
        row["window_start"] = (np.datetime64(center) - half_width).astype("datetime64[ns]")
        row["window_stop"] = (np.datetime64(center) + half_width).astype("datetime64[ns]")
        rows.append(row)

    out_time = np.asarray([row["window_center"] for row in rows], dtype="datetime64[ns]")

    def arr(name: str, dtype=float) -> np.ndarray:
        return np.asarray([row.get(name, np.nan) for row in rows], dtype=dtype)

    b0_values = (
        np.column_stack([arr("B0x_mso_nT"), arr("B0y_mso_nT"), arr("B0z_mso_nT")])
        if rows
        else np.empty((0, 3), dtype=float)
    )
    is_icw = arr("is_icw", dtype=bool)

    result = {
        "is_icw": is_icw,
        "is_icw_ts": ts_scalar(out_time, is_icw.astype(float), attrs={"name": "is_icw"}),
        "B0": ts_vec_xyz(out_time, b0_values, attrs={"name": "B0", "UNITS": "nT", "coordinates": "MSO"}),
        "B0_abs_nT": ts_scalar(out_time, arr("B0_abs_nT"), attrs={"name": "B0 magnitude", "UNITS": "nT"}),
        "fci_hz": ts_scalar(out_time, arr("fci_hz"), attrs={"name": "local gyrofrequency", "UNITS": "Hz"}),
        "Tci_s": ts_scalar(out_time, arr("Tci_s"), attrs={"name": "local gyroperiod", "UNITS": "s"}),
        "theta_kb_deg": ts_scalar(out_time, arr("theta_kb_deg"), attrs={"name": "theta_kB", "UNITS": "deg"}),
        "ellipticity": ts_scalar(out_time, arr("ellipticity"), attrs={"name": "ellipticity"}),
        "psd_bperp_0p6_0p8_fci": ts_scalar(out_time, arr("psd_bperp_0p6_0p8_fci")),
        "psd_bperp_0p8_1p2_fci": ts_scalar(out_time, arr("psd_bperp_0p8_1p2_fci")),
        "psd_bperp_1p2_1p4_fci": ts_scalar(out_time, arr("psd_bperp_1p2_1p4_fci")),
        "psd_bpara_0p8_1p2_fci": ts_scalar(out_time, arr("psd_bpara_0p8_1p2_fci")),
        "perp_main_over_low_ratio": ts_scalar(out_time, arr("perp_main_over_low_ratio")),
        "perp_main_over_high_ratio": ts_scalar(out_time, arr("perp_main_over_high_ratio")),
        "perp_over_para_main_ratio": ts_scalar(out_time, arr("perp_over_para_main_ratio")),
        "continuous_duration_s": ts_scalar(out_time, arr("continuous_duration_s"), attrs={"UNITS": "s"}),
        "required_duration_s": ts_scalar(out_time, arr("required_duration_s"), attrs={"UNITS": "s"}),
        "psd_ok": ts_scalar(out_time, arr("psd_ok", dtype=bool).astype(float)),
        "theta_ok": ts_scalar(out_time, arr("theta_ok", dtype=bool).astype(float)),
        "ellipticity_ok": ts_scalar(out_time, arr("ellipticity_ok", dtype=bool).astype(float)),
        "duration_ok": ts_scalar(out_time, arr("duration_ok", dtype=bool).astype(float)),
        "rows": rows,
        "time": out_time,
        "freqs_hz": np.asarray(wave_res["Bperp"].coords["frequency"].data, dtype=float),
        "wave_res": wave_res,
        "B0_all": B0_all,
        "duration_s": duration_s,
        "svd_window_length": svd_window_length,
        "svd_overlap": svd_overlap,
        "criteria": {
            "psd_main_over_low_min": sideband_ratio_min,
            "psd_main_over_high_min": sideband_ratio_min,
            "psd_perp_over_para_min": perp_para_ratio_min,
            "theta_kb_max_deg": theta_kb_max_deg,
            "ellipticity_max": ellipticity_max,
            "min_duration_gyroperiods": min_duration_gyroperiods,
        },
    }
    if plot:
        result["figure"] = plot_pcw_svd_psd_criteria(Bwave, result, tint_focus=tint_focus, freq_range=plot_freq_range)
    return result


def plot_pcw_svd_psd_criteria(
    Bwave: Any,
    result: dict[str, Any],
    *,
    tint_focus: tuple[str, str] | list[str] | tuple[np.datetime64, np.datetime64] | None = None,
    freq_range: tuple[float, float] = (0.01, 11.0),
) -> Any:
    """Plot B, SVD PSD, theta_kB, ellipticity, and PCW flag."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    b_time, b_data = _extract_bwave_arrays(Bwave)
    if tint_focus is None:
        focus_start, focus_stop = b_time[0], b_time[-1]
    else:
        focus_start, focus_stop = np.datetime64(tint_focus[0]), np.datetime64(tint_focus[1])
    bmask = (b_time >= focus_start) & (b_time <= focus_stop)
    svd_time = np.asarray(result["wave_res"]["Bperp"].time.data)
    smask = (svd_time >= focus_start) & (svd_time <= focus_stop)
    freq = np.asarray(result["wave_res"]["Bperp"].coords["frequency"].data, dtype=float)
    tnum = mdates.date2num(svd_time[smask].astype("datetime64[ms]").astype(object))

    fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    axes[0].plot(b_time[bmask].astype("datetime64[ms]").astype(object), b_data[bmask, 0], lw=0.7, color="tab:red", label="Bx")
    axes[0].plot(b_time[bmask].astype("datetime64[ms]").astype(object), b_data[bmask, 1], lw=0.7, color="tab:green", label="By")
    axes[0].plot(b_time[bmask].astype("datetime64[ms]").astype(object), b_data[bmask, 2], lw=0.7, color="tab:blue", label="Bz")
    axes[0].set_ylabel("B MSO\n(nT)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].grid(True, ls=":", alpha=0.35)

    panels = [
        ("Bperp PSD", result["wave_res"]["Bperp"], "Spectral_r", True),
        ("Bpara PSD", result["wave_res"]["Bpara"], "Spectral_r", True),
        (r"$\theta_{kB}$ (deg)", result["wave_res"]["theta"], "coolwarm", False),
        ("Ellipticity", result["wave_res"]["ellipticity"], "coolwarm", False),
    ]
    for ax, (label, da, cmap, use_log) in zip(axes[1:5], panels):
        data = np.asarray(da.values, dtype=float)[smask].T
        norm = None
        if use_log:
            positive = data[np.isfinite(data) & (data > 0)]
            norm = LogNorm(vmin=np.nanpercentile(positive, 2), vmax=np.nanpercentile(positive, 98)) if positive.size else None
        pc = ax.pcolormesh(tnum, freq, data, shading="auto", cmap=cmap, norm=norm)
        ax.set_yscale("log")
        ax.set_ylim(freq_range[0], freq_range[1])
        ax.set_ylabel("Freq\n(Hz)")
        ax.grid(True, which="both", ls=":", alpha=0.2)
        ax.text(0.01, 0.84, label, transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        fig.colorbar(pc, ax=ax, pad=0.01)

    centers = np.asarray(result["time"])
    centers_obj = centers.astype("datetime64[ms]").astype(object)
    fci = np.asarray(result["fci_hz"].data, dtype=float)
    blow = np.asarray([row.get("band_low_hz", np.nan) for row in result["rows"]], dtype=float)
    bhigh = np.asarray([row.get("band_high_hz", np.nan) for row in result["rows"]], dtype=float)
    for ax in axes[1:5]:
        ax.plot(centers_obj, fci, color="black", lw=2.0, label="fci")
        ax.plot(centers_obj, blow, color="black", lw=1.2, ls="--", label="0.8, 1.2 fci")
        ax.plot(centers_obj, bhigh, color="black", lw=1.2, ls="--")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[5].step(centers_obj, result["is_icw"].astype(float), where="mid", color="tab:red", lw=2.4)
    axes[5].set_ylabel("is PCW")
    axes[5].set_ylim(-0.1, 1.15)
    axes[5].set_yticks([0, 1])
    axes[5].grid(True, ls=":", alpha=0.35)
    axes[5].set_xlabel("Time")
    axes[5].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    for ax in axes:
        ax.set_xlim(focus_start.astype("datetime64[ms]").astype(object), focus_stop.astype("datetime64[ms]").astype(object))
    axes[0].set_title(
        f"PCW SVD PSD criteria, {_time_text(focus_start)} to {_time_text(focus_stop)}"
    )
    fig.autofmt_xdate()
    return fig


__all__ = [
    "detect_pcw_svd_psd_criteria",
    "evaluate_pcw_svd_psd_criteria_window",
    "plot_pcw_svd_psd_criteria",
    "read_sw_interval_list",
]

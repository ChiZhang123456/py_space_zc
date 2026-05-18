"""Romeo-style PCW detection using Welch PSD and MVA.

This module implements a practical version of the PCW selection described by
Romeo et al. (2020): transverse magnetic PSD near the local proton gyrofrequency
is compared with neighboring frequency bands, and MVA is used to estimate
planarity and handedness in subintervals of about three local gyroperiods.
"""

from __future__ import annotations

from typing import Any
import warnings

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


def _time_text(value: Any) -> str:
    return np.datetime_as_string(np.asarray(value).astype("datetime64[s]"), unit="s")


def _extract_bwave_arrays(Bwave: Any) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(Bwave, "time") or not hasattr(Bwave, "data"):
        raise TypeError("Bwave must be an xarray DataArray with time and 3-component data")
    Bclean = Bwave.dropna(dim="time") if hasattr(Bwave, "dropna") else Bwave
    time = np.asarray(Bclean.time.data).astype("datetime64[ns]")
    data = np.asarray(Bclean.data, dtype=float)
    if data.ndim != 2 or 3 not in data.shape:
        raise ValueError("Bwave data must be a 2D array with one dimension of length 3")
    if data.shape[0] == 3 and data.shape[1] != 3:
        data = data.T
    if data.shape[1] != 3:
        raise ValueError("cannot identify the 3-component dimension in Bwave")
    if len(time) != data.shape[0]:
        raise ValueError("Bwave time coordinate length does not match data length")
    good = np.all(np.isfinite(data), axis=1)
    return time[good], data[good]


def _sampling_rate_hz(time: np.ndarray) -> float:
    time = np.asarray(time).astype("datetime64[ns]")
    if time.size < 2:
        return np.nan
    dt_s = np.nanmedian(np.diff(time).astype("timedelta64[ns]").astype(float)) * 1.0e-9
    return float(1.0 / dt_s) if np.isfinite(dt_s) and dt_s > 0 else np.nan


def _fci_hz(b_abs_nt: float, ion: str | tuple[float, float]) -> float:
    mass, charge = _ion_mass_charge(ion)
    return abs(charge) * b_abs_nt * 1.0e-9 / (2.0 * np.pi * mass)


def _band_nanmean(freq: np.ndarray, values: np.ndarray, band: tuple[float, float]) -> float:
    mask = np.isfinite(freq) & np.isfinite(values) & (freq >= band[0]) & (freq <= band[1])
    return float(np.nanmean(values[mask])) if np.any(mask) else np.nan


def _make_vec(time: np.ndarray, data: np.ndarray, template: Any | None = None):
    from py_space_zc.ts_vec_xyz import ts_vec_xyz

    attrs = getattr(template, "attrs", {}) if template is not None else {}
    return ts_vec_xyz(np.asarray(time).astype("datetime64[ns]"), np.asarray(data, dtype=float), attrs=attrs)


def _make_scalar(time: np.ndarray, data: np.ndarray, name: str = "", units: str | None = None):
    from py_space_zc.ts_scalar import ts_scalar

    attrs: dict[str, str] = {}
    if name:
        attrs["name"] = name
    if units:
        attrs["UNITS"] = units
    return ts_scalar(np.asarray(time).astype("datetime64[ns]"), np.asarray(data), attrs=attrs)


def _welch_psd_fac(
    time: np.ndarray,
    b_xyz_nt: np.ndarray,
    fci_hz: float,
    B_template: Any | None,
    low_band_fci: tuple[float, float],
    main_band_fci: tuple[float, float],
    high_band_fci: tuple[float, float],
) -> dict[str, Any]:
    try:
        from scipy.signal import welch
    except ImportError as exc:
        raise ImportError("scipy is required for Welch PSD PCW criteria") from exc

    from py_space_zc.method.fac import fac

    if time.size < 16 or not np.isfinite(fci_hz) or fci_hz <= 0:
        return _empty_welch()

    fs = _sampling_rate_hz(time)
    if not np.isfinite(fs) or fs <= 0:
        return _empty_welch()

    B_da = _make_vec(time, b_xyz_nt, B_template)
    B_fac, *_ = fac(B_da)
    b_fac = np.asarray(B_fac.data, dtype=float)
    b_fac = b_fac - np.nanmean(b_fac, axis=0, keepdims=True)

    target_df = max(0.1 * fci_hz, 1.0 / max(time.size / fs, 1.0))
    nperseg = int(np.clip(np.ceil(fs / target_df), 16, b_fac.shape[0]))
    noverlap = nperseg // 2 if nperseg < b_fac.shape[0] else 0

    freq, p1 = welch(b_fac[:, 0], fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, p2 = welch(b_fac[:, 1], fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, ppara = welch(b_fac[:, 2], fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    pperp = p1 + p2

    low_band = (low_band_fci[0] * fci_hz, low_band_fci[1] * fci_hz)
    main_band = (main_band_fci[0] * fci_hz, main_band_fci[1] * fci_hz)
    high_band = (high_band_fci[0] * fci_hz, high_band_fci[1] * fci_hz)
    p_low = _band_nanmean(freq, pperp, low_band)
    p_main = _band_nanmean(freq, pperp, main_band)
    p_high = _band_nanmean(freq, pperp, high_band)
    p_para = _band_nanmean(freq, ppara, main_band)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_main_low = p_main / p_low
        ratio_main_high = p_main / p_high
        ratio_perp_para = p_main / p_para

    return {
        "freq": freq,
        "p_perp": pperp,
        "p_para": ppara,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "df_hz": float(freq[1] - freq[0]) if freq.size > 1 else np.nan,
        "low_band_hz": low_band,
        "main_band_hz": main_band,
        "high_band_hz": high_band,
        "psd_bperp_low": p_low,
        "psd_bperp_main": p_main,
        "psd_bperp_high": p_high,
        "psd_bpara_main": p_para,
        "ratio_main_low": ratio_main_low,
        "ratio_main_high": ratio_main_high,
        "ratio_perp_para": ratio_perp_para,
    }


def _empty_welch() -> dict[str, Any]:
    return {
        "freq": np.asarray([], dtype=float),
        "p_perp": np.asarray([], dtype=float),
        "p_para": np.asarray([], dtype=float),
        "nperseg": 0,
        "noverlap": 0,
        "df_hz": np.nan,
        "low_band_hz": (np.nan, np.nan),
        "main_band_hz": (np.nan, np.nan),
        "high_band_hz": (np.nan, np.nan),
        "psd_bperp_low": np.nan,
        "psd_bperp_main": np.nan,
        "psd_bperp_high": np.nan,
        "psd_bpara_main": np.nan,
        "ratio_main_low": np.nan,
        "ratio_main_high": np.nan,
        "ratio_perp_para": np.nan,
    }


def _mva_subintervals(
    time: np.ndarray,
    b_xyz_nt: np.ndarray,
    b0_vec_nt: np.ndarray,
    fci_hz: float,
    B_template: Any | None,
    mva_periods: float,
    mva_step_periods: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from py_space_zc.method.mva import mva

    if time.size < 16 or not np.isfinite(fci_hz) or fci_hz <= 0:
        return [], _empty_mva_summary()

    t0 = time[0]
    sec = (time - t0).astype("timedelta64[ns]").astype(float) * 1.0e-9
    duration_s = sec[-1] - sec[0]
    sub_len_s = mva_periods / fci_hz
    step_s = (mva_step_periods / fci_hz) if mva_step_periods is not None else sub_len_s
    if sub_len_s <= 0 or step_s <= 0 or duration_s < sub_len_s:
        return [], _empty_mva_summary()

    rows: list[dict[str, Any]] = []
    starts = np.arange(0.0, duration_s - sub_len_s + 1.0e-9, step_s)
    for start_s in starts:
        stop_s = start_s + sub_len_s
        mask = (sec >= start_s) & (sec < stop_s)
        if np.count_nonzero(mask) < 16:
            continue
        sub_time = time[mask]
        sub_b = b_xyz_nt[mask]
        b_mva, lamb, frame = mva(_make_vec(sub_time, sub_b, B_template))
        lamb = np.asarray(lamb, dtype=float)
        frame = np.asarray(frame, dtype=float)
        b_mva_data = np.asarray(b_mva.data, dtype=float)
        b0_mva = frame.T @ b0_vec_nt

        centered = b_mva_data - np.nanmean(b_mva_data, axis=0, keepdims=True)
        rot_e3 = np.nanmean(centered[:-1, 0] * centered[1:, 1] - centered[:-1, 1] * centered[1:, 0])
        p_s = np.sign(rot_e3 * b0_mva[2])
        if p_s == 0:
            p_s = np.nan

        e3 = frame[:, 2]
        cos_kb = abs(float(np.dot(e3, b0_vec_nt))) / (np.linalg.norm(e3) * np.linalg.norm(b0_vec_nt))
        theta_kb = np.degrees(np.arccos(np.clip(cos_kb, -1.0, 1.0)))
        lambda12 = lamb[0] / lamb[1] if lamb[1] > 0 else np.nan
        lambda23 = lamb[1] / lamb[2] if lamb[2] > 0 else np.nan
        delta_b = np.sqrt(max(lamb[0] - lamb[2], 0.0))

        rows.append(
            {
                "sub_start": sub_time[0].astype("datetime64[ns]"),
                "sub_stop": sub_time[-1].astype("datetime64[ns]"),
                "lambda1": float(lamb[0]),
                "lambda2": float(lamb[1]),
                "lambda3": float(lamb[2]),
                "lambda12": float(lambda12) if np.isfinite(lambda12) else np.nan,
                "lambda23": float(lambda23) if np.isfinite(lambda23) else np.nan,
                "theta_kb_deg": float(theta_kb) if np.isfinite(theta_kb) else np.nan,
                "deltaB_nT": float(delta_b) if np.isfinite(delta_b) else np.nan,
                "rot_e3": float(rot_e3) if np.isfinite(rot_e3) else np.nan,
                "B0_mva_e3_nT": float(b0_mva[2]) if np.isfinite(b0_mva[2]) else np.nan,
                "p_s": float(p_s) if np.isfinite(p_s) else np.nan,
            }
        )

    if not rows:
        return rows, _empty_mva_summary()

    def mean(key: str) -> float:
        values = np.asarray([row[key] for row in rows], dtype=float)
        return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan

    summary = {
        "mva_count": int(len(rows)),
        "lambda12_mean": mean("lambda12"),
        "lambda23_mean": mean("lambda23"),
        "theta_kb_mean_deg": mean("theta_kb_deg"),
        "deltaB_mean_nT": mean("deltaB_nT"),
        "polarization_p": mean("p_s"),
    }
    return rows, summary


def _empty_mva_summary() -> dict[str, Any]:
    return {
        "mva_count": 0,
        "lambda12_mean": np.nan,
        "lambda23_mean": np.nan,
        "theta_kb_mean_deg": np.nan,
        "deltaB_mean_nT": np.nan,
        "polarization_p": np.nan,
    }


def _window_bounds(time: np.ndarray, window_s: float, overlap: float) -> list[tuple[np.datetime64, np.datetime64]]:
    start = time[0]
    stop = time[-1]
    duration_s = float((stop - start) / np.timedelta64(1, "s"))
    if duration_s <= window_s:
        return [(start, stop)]
    if overlap < 1.0:
        step_s = window_s * (1.0 - overlap)
    else:
        step_s = window_s - overlap
    if step_s <= 0:
        raise ValueError("overlap must leave a positive step between windows")
    bounds: list[tuple[np.datetime64, np.datetime64]] = []
    for start_s in np.arange(0.0, duration_s - window_s + 1.0e-9, step_s):
        w0 = start + np.rint(start_s * 1.0e9).astype("timedelta64[ns]")
        w1 = w0 + np.rint(window_s * 1.0e9).astype("timedelta64[ns]")
        bounds.append((w0.astype("datetime64[ns]"), w1.astype("datetime64[ns]")))
    return bounds


def detect_pcw_welch_mva(
    Bwave: Any,
    ion: str | tuple[float, float] = "H+",
    *,
    window_s: float = 512.0,
    overlap: float = 0.90,
    min_duration_s: float = 180.0,
    low_band_fci: tuple[float, float] = (0.5, 0.7),
    main_band_fci: tuple[float, float] = (0.75, 1.15),
    high_band_fci: tuple[float, float] = (1.2, 1.4),
    main_over_low_min: float = 1.0,
    main_over_high_min: float = 1.5,
    perp_over_para_min: float = 3.0,
    lambda23_min: float = 5.0,
    require_left_hand: bool = True,
    mva_periods: float = 3.0,
    mva_step_periods: float | None = None,
    plot: bool = False,
) -> dict[str, Any]:
    """Detect PCW intervals with Welch PSD and MVA criteria.

    Parameters
    ----------
    Bwave
        xarray.DataArray with dimensions ``time`` and three magnetic-field
        components. Units are assumed to be nT.
    ion
        Ion species for the local gyrofrequency. Default is H+.
    window_s
        Main Welch/MVA averaging window length in seconds. If the input
        interval is shorter than this, the full input interval is evaluated
        as one window, as long as it is longer than ``min_duration_s``.
    overlap
        If less than 1, interpreted as fractional overlap. The Romeo-style
        default is 0.90.
    min_duration_s
        Warn when the total input duration is shorter than this value.
    low_band_fci, main_band_fci, high_band_fci
        Frequency bands normalized by the local gyrofrequency.
    plot
        If True, attach a diagnostic matplotlib figure as ``result["figure"]``.

    Returns
    -------
    result
        Dictionary containing per-window rows, xarray time series, and
        optional diagnostic figure.
    """
    time, b_xyz = _extract_bwave_arrays(Bwave)
    if time.size < 2:
        return {"rows": [], "is_pcw": np.asarray([], dtype=bool), "reason": "too_few_samples"}

    sample_dt_s = np.nanmedian(np.diff(time).astype("timedelta64[ns]").astype(float)) * 1.0e-9
    duration_s = float((time[-1] - time[0]) / np.timedelta64(1, "s")) + float(sample_dt_s)
    if duration_s < min_duration_s:
        warnings.warn(
            f"Bwave duration is {duration_s:.1f} s, shorter than the recommended {min_duration_s:.1f} s.",
            RuntimeWarning,
            stacklevel=2,
        )

    rows: list[dict[str, Any]] = []
    mva_subintervals: list[list[dict[str, Any]]] = []
    psd_products: list[dict[str, Any]] = []
    for window_id, (w0, w1) in enumerate(_window_bounds(time, window_s, overlap), start=1):
        mask = (time >= w0) & (time < w1)
        if np.count_nonzero(mask) < 16:
            continue
        w_time = time[mask]
        w_b = b_xyz[mask]
        b0_vec = np.nanmean(w_b, axis=0)
        b0_abs = float(np.linalg.norm(b0_vec))
        fci = _fci_hz(b0_abs, ion)

        psd = _welch_psd_fac(w_time, w_b, fci, Bwave, low_band_fci, main_band_fci, high_band_fci)
        sub_rows, mva_summary = _mva_subintervals(
            w_time, w_b, b0_vec, fci, Bwave, mva_periods, mva_step_periods
        )
        psd_products.append(psd)
        mva_subintervals.append(sub_rows)

        psd_ok = bool(
            np.isfinite(psd["ratio_main_low"])
            and np.isfinite(psd["ratio_main_high"])
            and np.isfinite(psd["ratio_perp_para"])
            and psd["ratio_main_low"] > main_over_low_min
            and psd["ratio_main_high"] > main_over_high_min
            and psd["ratio_perp_para"] > perp_over_para_min
        )
        mva_planar_ok = bool(np.isfinite(mva_summary["lambda23_mean"]) and mva_summary["lambda23_mean"] > lambda23_min)
        left_hand_ok = bool(
            np.isfinite(mva_summary["polarization_p"]) and mva_summary["polarization_p"] < 0.0
        )
        is_pcw = bool(psd_ok and mva_planar_ok and (left_hand_ok or not require_left_hand))

        rows.append(
            {
                "window_id": window_id,
                "window_start": w0.astype("datetime64[ns]"),
                "window_stop": w1.astype("datetime64[ns]"),
                "window_center": (w0 + (w1 - w0) // 2).astype("datetime64[ns]"),
                "B0x_nT": float(b0_vec[0]),
                "B0y_nT": float(b0_vec[1]),
                "B0z_nT": float(b0_vec[2]),
                "B0_abs_nT": b0_abs,
                "fci_hz": float(fci),
                "Tci_s": float(1.0 / fci) if fci > 0 else np.nan,
                "low_band_low_hz": float(psd["low_band_hz"][0]),
                "low_band_high_hz": float(psd["low_band_hz"][1]),
                "main_band_low_hz": float(psd["main_band_hz"][0]),
                "main_band_high_hz": float(psd["main_band_hz"][1]),
                "high_band_low_hz": float(psd["high_band_hz"][0]),
                "high_band_high_hz": float(psd["high_band_hz"][1]),
                "psd_bperp_low": float(psd["psd_bperp_low"]) if np.isfinite(psd["psd_bperp_low"]) else np.nan,
                "psd_bperp_main": float(psd["psd_bperp_main"]) if np.isfinite(psd["psd_bperp_main"]) else np.nan,
                "psd_bperp_high": float(psd["psd_bperp_high"]) if np.isfinite(psd["psd_bperp_high"]) else np.nan,
                "psd_bpara_main": float(psd["psd_bpara_main"]) if np.isfinite(psd["psd_bpara_main"]) else np.nan,
                "main_over_low_ratio": float(psd["ratio_main_low"]) if np.isfinite(psd["ratio_main_low"]) else np.nan,
                "main_over_high_ratio": float(psd["ratio_main_high"]) if np.isfinite(psd["ratio_main_high"]) else np.nan,
                "perp_over_para_ratio": float(psd["ratio_perp_para"]) if np.isfinite(psd["ratio_perp_para"]) else np.nan,
                "welch_nperseg": int(psd["nperseg"]),
                "welch_df_hz": float(psd["df_hz"]) if np.isfinite(psd["df_hz"]) else np.nan,
                **mva_summary,
                "psd_ok": psd_ok,
                "mva_planar_ok": mva_planar_ok,
                "left_hand_ok": left_hand_ok,
                "is_pcw": is_pcw,
                "detection_logic": "welch_psd_fac_and_mva",
            }
        )

    out_time = np.asarray([row["window_center"] for row in rows], dtype="datetime64[ns]")

    def arr(name: str, dtype=float) -> np.ndarray:
        return np.asarray([row.get(name, np.nan) for row in rows], dtype=dtype)

    result = {
        "rows": rows,
        "mva_subintervals": mva_subintervals,
        "psd_products": psd_products,
        "time": out_time,
        "is_pcw": arr("is_pcw", dtype=bool),
        "is_pcw_ts": _make_scalar(out_time, arr("is_pcw", dtype=bool).astype(float), "is_pcw"),
        "B0": _make_vec(out_time, np.column_stack([arr("B0x_nT"), arr("B0y_nT"), arr("B0z_nT")]), Bwave),
        "B0_abs_nT": _make_scalar(out_time, arr("B0_abs_nT"), "B0_abs", "nT"),
        "fci_hz": _make_scalar(out_time, arr("fci_hz"), "fci", "Hz"),
        "main_over_low_ratio": _make_scalar(out_time, arr("main_over_low_ratio"), "main_over_low_ratio"),
        "main_over_high_ratio": _make_scalar(out_time, arr("main_over_high_ratio"), "main_over_high_ratio"),
        "perp_over_para_ratio": _make_scalar(out_time, arr("perp_over_para_ratio"), "perp_over_para_ratio"),
        "lambda23_mean": _make_scalar(out_time, arr("lambda23_mean"), "lambda23_mean"),
        "polarization_p": _make_scalar(out_time, arr("polarization_p"), "polarization_p"),
        "duration_s": duration_s,
        "criteria": {
            "low_band_fci": low_band_fci,
            "main_band_fci": main_band_fci,
            "high_band_fci": high_band_fci,
            "main_over_low_min": main_over_low_min,
            "main_over_high_min": main_over_high_min,
            "perp_over_para_min": perp_over_para_min,
            "lambda23_min": lambda23_min,
            "require_left_hand": require_left_hand,
            "mva_periods": mva_periods,
        },
    }
    if plot:
        result["figure"] = plot_pcw_welch_mva(Bwave, result)
    return result


def plot_pcw_welch_mva(Bwave: Any, result: dict[str, Any]) -> Any:
    """Plot the Romeo-style Welch and MVA criteria."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    time, b_xyz = _extract_bwave_arrays(Bwave)
    fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    t_obj = time.astype("datetime64[ms]").astype(object)
    axes[0].plot(t_obj, b_xyz[:, 0], lw=0.7, label="Bx", color="tab:red")
    axes[0].plot(t_obj, b_xyz[:, 1], lw=0.7, label="By", color="tab:green")
    axes[0].plot(t_obj, b_xyz[:, 2], lw=0.7, label="Bz", color="tab:blue")
    axes[0].set_ylabel("B\n(nT)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].grid(True, ls=":", alpha=0.35)

    centers = np.asarray(result["time"]).astype("datetime64[ms]").astype(object)
    axes[1].plot(centers, result["main_over_low_ratio"].data, marker="o", ms=3, label="main / low")
    axes[1].plot(centers, result["main_over_high_ratio"].data, marker="o", ms=3, label="main / high")
    axes[1].plot(centers, result["perp_over_para_ratio"].data, marker="o", ms=3, label="perp / para")
    axes[1].axhline(result["criteria"]["main_over_low_min"], color="0.4", ls="--", lw=1.0)
    axes[1].axhline(result["criteria"]["perp_over_para_min"], color="0.2", ls=":", lw=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Welch PSD\nratios")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].grid(True, which="both", ls=":", alpha=0.35)

    axes[2].plot(centers, result["lambda23_mean"].data, marker="o", ms=3, label="lambda2/lambda3")
    axes[2].axhline(result["criteria"]["lambda23_min"], color="0.2", ls="--", lw=1.0)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("MVA\nplanarity")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, which="both", ls=":", alpha=0.35)

    theta = np.asarray([row.get("theta_kb_mean_deg", np.nan) for row in result["rows"]], dtype=float)
    axes[3].plot(centers, result["polarization_p"].data, marker="o", ms=3, label="mean p")
    axes[3].plot(centers, theta, marker="o", ms=3, label="theta kB")
    axes[3].axhline(0.0, color="0.2", ls="--", lw=1.0)
    axes[3].set_ylabel("p, theta")
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)
    axes[3].grid(True, ls=":", alpha=0.35)

    axes[4].step(centers, result["is_pcw"].astype(float), where="mid", lw=2.2, color="tab:red", label="is_pcw")
    axes[4].set_ylim(-0.1, 1.15)
    axes[4].set_yticks([0, 1])
    axes[4].set_ylabel("PCW")
    axes[4].set_xlabel("Time")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].grid(True, ls=":", alpha=0.35)
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    for row in result["rows"]:
        if row.get("is_pcw", False):
            w0 = row["window_start"].astype("datetime64[ms]").astype(object)
            w1 = row["window_stop"].astype("datetime64[ms]").astype(object)
            for ax in axes:
                ax.axvspan(w0, w1, color="tab:red", alpha=0.08)

    axes[0].set_title(
        "Romeo-style PCW detection: Welch PSD in FAC plus MVA"
    )
    fig.autofmt_xdate()
    return fig


__all__ = ["detect_pcw_welch_mva", "plot_pcw_welch_mva"]

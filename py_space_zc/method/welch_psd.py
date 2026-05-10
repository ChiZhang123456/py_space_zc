import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from py_space_zc import ts_spectr, ts_vec_xyz
from py_space_zc.plot import apply_plot_font, configure_plot_font
from .fac import fac

def _time_seconds_from_datetime64(time_raw):
    time_raw = np.asarray(time_raw)
    return (time_raw - time_raw[0]) / np.timedelta64(1, "s")


def _linear_fit_loglog(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return slope, intercept, np.sum(residual**2)


def _prefix_sums(x, y):
    return {
        "n": np.arange(x.size + 1, dtype=float),
        "x": np.concatenate(([0.0], np.cumsum(x))),
        "y": np.concatenate(([0.0], np.cumsum(y))),
        "xx": np.concatenate(([0.0], np.cumsum(x * x))),
        "xy": np.concatenate(([0.0], np.cumsum(x * y))),
        "yy": np.concatenate(([0.0], np.cumsum(y * y))),
    }


def _range_sum(prefix, key, start, end):
    return prefix[key][end] - prefix[key][start]


def _segment_fit_from_prefix(prefix, start, end):
    n = end - start
    sx = _range_sum(prefix, "x", start, end)
    sy = _range_sum(prefix, "y", start, end)
    sxx = _range_sum(prefix, "xx", start, end)
    sxy = _range_sum(prefix, "xy", start, end)
    syy = _range_sum(prefix, "yy", start, end)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        slope = 0.0
        intercept = sy / n
    else:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

    rss = syy - slope * sxy - intercept * sy
    return slope, intercept, max(float(rss), 0.0)


def _side_median_baseline(y, side_window=24, guard_points=2):
    baseline = np.full(y.size, np.nan, dtype=float)
    for i in range(y.size):
        left = y[max(0, i - side_window) : max(0, i - guard_points)]
        right = y[min(y.size, i + guard_points + 1) : min(y.size, i + side_window + 1)]
        neighbors = np.concatenate((left, right))
        neighbors = neighbors[np.isfinite(neighbors)]
        if neighbors.size >= 3:
            baseline[i] = np.median(neighbors)

    missing = ~np.isfinite(baseline)
    if np.any(missing):
        fallback = np.nanmedian(y)
        baseline[missing] = fallback
    return baseline


def _fit_spike_mask(log_psd, max_factor=4.0):
    """Return False for high narrow peaks that should not control the fit."""
    log_psd = np.asarray(log_psd, dtype=float)
    keep = np.isfinite(log_psd)
    if np.count_nonzero(keep) < 8:
        return keep

    baseline = _side_median_baseline(log_psd)
    keep &= (log_psd - baseline) <= np.log10(max_factor)
    return keep


def _fit_segmented_loglog(x, y, f_fit, n_segments, min_points):
    n = x.size
    if n_segments < 1 or n_segments > 4:
        raise ValueError("n_segments must be 1, 2, 3, or 4")
    if n < n_segments * min_points:
        return None

    prefix = _prefix_sums(x, y)
    dp = np.full((n_segments + 1, n + 1), np.inf)
    prev = np.full((n_segments + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, n_segments + 1):
        end_min = k * min_points
        for end in range(end_min, n + 1):
            start_min = (k - 1) * min_points
            start_max = end - min_points
            best_cost = np.inf
            best_start = -1
            for start in range(start_min, start_max + 1):
                if not np.isfinite(dp[k - 1, start]):
                    continue
                _, _, rss = _segment_fit_from_prefix(prefix, start, end)
                cost = dp[k - 1, start] + rss
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
            dp[k, end] = best_cost
            prev[k, end] = best_start

    if not np.isfinite(dp[n_segments, n]):
        return None

    edges = [n]
    end = n
    for k in range(n_segments, 0, -1):
        start = prev[k, end]
        if start < 0:
            return None
        edges.append(start)
        end = start
    edges = list(reversed(edges))

    segments = []
    for start, end in zip(edges[:-1], edges[1:]):
        slope, intercept, rss = _segment_fit_from_prefix(prefix, start, end)
        segments.append({"start": start, "end": end, "slope": slope, "intercept": intercept, "rss": rss})

    break_indices = edges[1:-1]
    return {
        "rss": float(dp[n_segments, n]),
        "segments": segments,
        "break_indices": break_indices,
        "f_breaks": [f_fit[idx] for idx in break_indices],
    }


def fit_psd_break(
    f,
    psd,
    fit_range=(0.01, 5.0),
    fit_mode="auto",
    min_points=6,
    rss_reduction_threshold=0.30,
):
    """
    Fit power-law PSD slopes in log-log space with up to 4 segments.

    Parameters:
    -----------
    fit_mode: "1", "2", "3", "4", or "auto"
    rss_reduction_threshold: 0.30 (30% improvement needed to accept a more complex model)
    """
    aliases = {
        "auto": "auto", 
        "1": "single", 
        "2": "double", 
        "3": "triple", 
        "4": "quad"
    }
    
    if str(fit_mode) not in aliases:
        raise ValueError('fit_mode must be "auto", "1", "2", "3", or "4"')

    # 1. Data Validation and Range Selection
    f = np.asarray(f, dtype=float)
    psd = np.asarray(psd, dtype=float)
    valid = (
        np.isfinite(f) & np.isfinite(psd) & 
        (f > 0) & (psd > 0) & 
        (f >= fit_range[0]) & (f <= fit_range[1])
    )

    f_fit = f[valid]
    psd_fit = psd[valid]
    order = np.argsort(f_fit)
    f_fit, psd_fit = f_fit[order], psd_fit[order]

    if f_fit.size < 2 * min_points:
        return {"accepted": False, "reason": "not_enough_points", "selected": None}

    # 2. Log-Space Transformation and Spike Masking
    x_all = np.log10(f_fit)
    y_all = np.log10(psd_fit)
    keep_fit = _fit_spike_mask(y_all)
    
    x, y = x_all[keep_fit], y_all[keep_fit]
    f_used, psd_used = f_fit[keep_fit], psd_fit[keep_fit]

    # Fallback if masking removes too many points
    if f_used.size < 2 * min_points:
        x, y, f_used, psd_used = x_all, y_all, f_fit, psd_fit

    # 3. Calculate All Models
    single = _fit_segmented_loglog(x, y, f_used, 1, min_points)
    double = _fit_segmented_loglog(x, y, f_used, 2, min_points)
    triple = _fit_segmented_loglog(x, y, f_used, 3, min_points)
    quad   = _fit_segmented_loglog(x, y, f_used, 4, min_points)

    # 4. Format Results (Legacy structures for compatibility)
    def _wrap_result(res, n):
        if res is None: return None
        improvement = 1.0 - res["rss"] / single["rss"] if single["rss"] > 0 else 0.0
        out = {
            "rss": res["rss"],
            "rss_reduction": improvement,
            "segments": res["segments"],
            "f_breaks": res["f_breaks"],
        }
        for i in range(n):
            out[f"slope_{i+1}"] = res["segments"][i]["slope"]
        if n > 1: out["f_break"] = res["f_breaks"][0]
        return out

    single_res = _wrap_result(single, 1)
    double_res = _wrap_result(double, 2)
    triple_res = _wrap_result(triple, 3)
    quad_res   = _wrap_result(quad, 4)

    # 5. Selection Logic
    mode = aliases[str(fit_mode)]
    accepted = True
    reason = "forced_selection"

    if mode == "single":
        selected_mode, selected = "single", single_res
    elif mode == "double":
        selected_mode, selected = "double", double_res
    elif mode == "triple":
        selected_mode, selected = "triple", triple_res
    elif mode == "quad":
        selected_mode, selected = "quad", quad_res
    else: # "auto"
        # Preference: 2 segments is usually the standard for "break" detection
        if double_res and double_res["rss_reduction"] >= rss_reduction_threshold:
            selected_mode, selected, reason = "double", double_res, "accepted_auto"
        else:
            selected_mode, selected, reason = "single", single_res, "rss_reduction_below_threshold"
            accepted = False

    if selected is None:
        selected_mode, selected = "single", single_res
        accepted = False
        reason = "fallback_single"

    return {
        "accepted": accepted,
        "reason": reason,
        "selected_mode": selected_mode,
        "f_fit_used": f_used,
        "psd_fit_used": psd_used,
        "single": single_res,
        "double": double_res,
        "triple": triple_res,
        "quad": quad_res,
        "selected": selected,
    }


def _power_law_from_logfit(f, slope, intercept):
    return 10 ** (intercept + slope * np.log10(f))


def _first_crossing_time(tau, acf, level=1 / np.e):
    below = np.where(np.isfinite(acf) & (acf <= level))[0]
    return tau[below[0]] if below.size else np.nan


def _fill_nan_1d(series):
    series = np.asarray(series, dtype=float).copy()
    finite = np.isfinite(series)
    if finite.all():
        return series
    if not finite.any():
        return series
    idx = np.arange(series.size)
    series[~finite] = np.interp(idx[~finite], idx[finite], series[finite])
    return series


def _idl_smooth_edge_truncate(x, window):
    x = np.asarray(x, dtype=float)
    window = max(1, int(round(window)))
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(x).astype(float)
    values = np.where(np.isfinite(x), x, 0.0)
    smoothed = np.convolve(values, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")
    out = np.divide(smoothed, counts, out=np.full_like(x, np.nan), where=counts > 0)
    return out


def _rotate_to_fac_idl_for_acf(b_vec, window):
    b_vec = np.asarray(b_vec, dtype=float)
    b0 = np.column_stack([_idl_smooth_edge_truncate(b_vec[:, i], window) for i in range(3)])
    b0mag = np.linalg.norm(b0, axis=1)
    b0mag = np.maximum(b0mag, 1e-12)
    e_par = b0 / b0mag[:, None]

    ve = np.zeros_like(e_par)
    abs_b0 = np.abs(b0)
    ve[np.argmin(abs_b0, axis=1) == 0, 0] = 1.0
    ve[np.argmin(abs_b0, axis=1) == 1, 1] = 1.0
    ve[np.argmin(abs_b0, axis=1) == 2, 2] = 1.0

    e_perp2 = np.cross(e_par, ve)
    e_perp2 /= np.maximum(np.linalg.norm(e_perp2, axis=1), 1e-12)[:, None]
    e_perp1 = np.cross(e_perp2, e_par)
    e_perp1 /= np.maximum(np.linalg.norm(e_perp1, axis=1), 1e-12)[:, None]

    return np.column_stack(
        (
            np.sum(b_vec * e_par, axis=1),
            np.sum(b_vec * e_perp1, axis=1),
            np.sum(b_vec * e_perp2, axis=1),
        )
    )


def calculate_normalized_acf(series, dt, max_lags=None):
    """
    Normalized ACF:
        A_L = N/(N-L) * sum_{n=1}^{N-L} X_n X_{n+L} / sum_{n=1}^{N} X_n^2

    This follows the IDL autocorr_norm.pro definition exactly. The input is
    not de-meaned inside this helper; pass fluctuations or detrended data if
    the mean field should be removed before correlation.
    """
    x = _fill_nan_1d(series)
    if not np.isfinite(x).any():
        lags = np.arange(0 if max_lags is None else max_lags)
        return lags * dt, np.full(lags.size, np.nan)

    n = x.size
    if max_lags is None:
        max_lags = n
    max_lags = max(1, min(int(max_lags), n))
    lags = np.arange(max_lags)

    denom = np.sum(x * x)
    if not np.isfinite(denom) or denom <= 0:
        return lags * dt, np.full(max_lags, np.nan)

    corr = signal.correlate(x, x, mode="full", method="fft")
    acf = corr[n - 1 : n - 1 + max_lags] / denom
    acf *= n / (n - lags)
    acf[0] = 1.0
    return lags * dt, acf



def _auto_nperseg(n_total):
    target = max(1, int(np.floor(n_total / 8)))
    nperseg = 1 << int(np.ceil(np.log2(target)))
    return min(max(nperseg, 2), n_total)


def psd_welch(
    Bwave,
    plot=True,
    tint=None,
    fit_range=(0.01, 5.0),
    fit_component="total",
    fit_mode="auto",
):
    """
    Calculate PSD, magnetic compressibility, ACF, and spectral slopes/breaks.

    The figure shows the raw Welch PSD. The fit ignores obvious high narrow
    spikes internally, but the PSD arrays are not modified.

    Parameters
    ----------
    Welch segment length is selected automatically as
    2^nextpow2(floor(n_total / 8)), where n_total is the number of samples in
    the current interval.
    fit_range : tuple
        Frequency range used for fitting, in Hz.
    fit_component : {"total", "para", "perp"}
        PSD component used for fitting. "total" fits p_para + p_perp.
    fit_mode : {"auto", "1", "2", "3"}
        "auto": single-vs-double with a 30% RSS-reduction criterion.
        "1", "2", "3": force one, two, or three fitted power-law segments.
    """
    Bfac, fac_x, fac_y, fac_z, B_bgd = fac(Bwave)
    data = Bwave.data.copy()
    time_raw = Bwave.time.data

    if np.isnan(data).any():
        indices = np.arange(len(data))
        for i in range(3):
            mask = np.isnan(data[:, i])
            if mask.any():
                data[mask, i] = np.interp(indices[mask], indices[~mask], data[~mask, i])

    time_sec = _time_seconds_from_datetime64(time_raw)
    dt = np.median(np.diff(time_sec))
    fs = 1.0 / dt

    Bbgd_vec = np.nanmean(B_bgd.data.copy(), axis = 0)
    B_mag = np.linalg.norm(Bbgd_vec)
    B_fac = Bfac.data.copy()
    dB_fac = B_fac - np.nanmean(B_fac, axis=0)

    f_ci = 0.01524 * B_mag
    T_ci = 1.0 / f_ci if f_ci > 0 else np.nan

    n_actual = _auto_nperseg(dB_fac.shape[0])
    welch_kwargs = dict(
        fs=fs,
        nperseg=n_actual,
        noverlap=n_actual // 2,
        window="hann",
        average="median",
    )
    f, p_para = signal.welch(dB_fac[:, 2], **welch_kwargs)
    _, p_perp1 = signal.welch(dB_fac[:, 0], **welch_kwargs)
    _, p_perp2 = signal.welch(dB_fac[:, 1], **welch_kwargs)

    p_total = p_para + p_perp1 + p_perp2
    comp_ratio = p_para / (p_total)

    fit_targets = {"para": p_para, "perp": p_perp1 + p_perp2, "total": p_total}
    if fit_component not in fit_targets:
        raise ValueError('fit_component must be "total", "para", or "perp"')

    fit_result = fit_psd_break(
        f,
        fit_targets[fit_component],
        fit_range=fit_range,
        fit_mode=fit_mode,
    )
    acf_fac = _rotate_to_fac_idl_for_acf(data, window=max(1, round(60.0 / dt)))
    tau_axis_l, acf_l = calculate_normalized_acf(signal.detrend(acf_fac[:, 1], type="linear"), dt)
    tau_axis_m, acf_m = calculate_normalized_acf(signal.detrend(acf_fac[:, 2], type="linear"), dt)
    tau_axis_n, acf_n = calculate_normalized_acf(signal.detrend(acf_fac[:, 0], type="linear"), dt)

    if plot:
        configure_plot_font()
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
        ax_comp_psd, ax_total_psd = axes[0]
        ax_comp_ratio, ax_acf = axes[1]

        ax_comp_psd.loglog(f, p_para, label=r"$\delta B_{\parallel}$ (Para)", color="#1f77b4", lw=1.2)
        ax_comp_psd.loglog(f, p_perp1, label=r"$\delta B_{\perp 1}$ (Perp 1)", color="#ff7f0e", lw=1.2)
        ax_comp_psd.loglog(f, p_perp2, label=r"$\delta B_{\perp 2}$ (Perp 2)", color="#2ca02c", lw=1.2)

        ax_comp_psd.axvline(f_ci, color="red", ls="--", alpha=0.7, label=f"$f_{{ci}}$={f_ci:.2f} Hz")
        ax_comp_psd.set_xlabel("Frequency (Hz)", fontsize=13)
        ax_comp_psd.set_ylabel(r"PSD ($nT^2/Hz$)", fontsize=13)
        ax_comp_psd.grid(True, which="both", ls=":", alpha=0.5)
        ax_comp_psd.legend(fontsize=10)
        ax_comp_psd.set_title("Parallel and Perpendicular PSD", fontweight="bold", fontsize=14)

        ax_total_psd.loglog(f, p_total, label=r"$\delta B_{trace}$", color="black", lw=1.1)
        ax_total_psd.axvline(f_ci, color="red", ls="--", alpha=0.7, label=f"$f_{{ci}}$={f_ci:.2f} Hz")

        if fit_result["selected"] is not None:
            selected = fit_result["selected"]
            fit_color = "tab:green"
            for i, seg in enumerate(selected["segments"], start=1):
                f_seg = fit_result["f_fit_used"][seg["start"] : seg["end"]]
                y_seg = _power_law_from_logfit(f_seg, seg["slope"], seg["intercept"])
                label = f"{fit_result['selected_mode']}-fit" if i == 1 else None
                ax_total_psd.loglog(f_seg, y_seg, color=fit_color, lw=2.0, label=label)

            for i, f_break in enumerate(selected.get("f_breaks", []), start=1):
                ax_total_psd.axvline(
                    f_break, color=fit_color, ls="--", lw=1.2, label=rf"$f_{{b{i}}}$={f_break:.3g} Hz"
                )

            slopes = [seg["slope"] for seg in selected["segments"]]
            slope_text = ", ".join(rf"$\alpha_{i}$={slope:.2f}" for i, slope in enumerate(slopes, start=1))
            if "rss_reduction" in selected:
                slope_text += "\n" + rf"RSS drop={100 * selected['rss_reduction']:.0f}%"
            ax_total_psd.text(
                0.04,
                0.05,
                slope_text,
                transform=ax_total_psd.transAxes,
                fontsize=11,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.85),
            )

        ax_total_psd.set_xlabel("Frequency (Hz)", fontsize=13)
        ax_total_psd.set_ylabel(r"PSD ($nT^2/Hz$)", fontsize=13)
        ax_total_psd.grid(True, which="both", ls=":", alpha=0.5)
        ax_total_psd.legend(fontsize=10)
        ax_total_psd.set_title("Trace PSD and Power-Law Fit", fontweight="bold", fontsize=14)

        ax_comp_ratio.semilogx(f, comp_ratio, color="purple", lw=1.2)
        ax_comp_ratio.axvline(f_ci, color="red", ls="--", alpha=0.7)
        ax_comp_ratio.axhline(1/3, color="gray", ls="--", label="Isotropic (1/3)")
        if fit_result["selected"] is not None:
            for f_break in fit_result["selected"].get("f_breaks", []):
                ax_comp_ratio.axvline(f_break, color="tab:green", ls="--", lw=1.2)
        ax_comp_ratio.set_xlabel("Frequency (Hz)", fontsize=13)
        ax_comp_ratio.set_ylabel(r"$C_{\parallel} = PSD_{\parallel} / PSD_{Total}$", fontsize=13)
        ax_comp_ratio.set_ylim(0, 1.0)
        ax_comp_ratio.grid(True, which="both", ls=":", alpha=0.5)
        ax_comp_ratio.legend(fontsize=10)
        ax_comp_ratio.set_title("Magnetic Compressibility", fontweight="bold", fontsize=14)


        ax_acf.plot(tau_axis_l, acf_l, label=r"ACF $\delta B_{\perp 1}$", color="#1f77b4")
        ax_acf.plot(tau_axis_m, acf_m, label=r"ACF $\delta B_{\perp 2}$", color="#2ca02c", alpha=0.7)
        ax_acf.plot(tau_axis_n, acf_n, label=r"ACF $\delta B_{\parallel}$", color="#d62728", alpha=0.7)
        # ax_acf.axvline(T_ci, color="red", ls="--", alpha=0.7, label=f"$T_{{ci}}$={T_ci:.2f}s")
        ax_acf.axhline(1 / np.e, color="gray", ls=":", label="1/e")
        ax_acf.set_xlabel("Time Lag $L$ (s)", fontsize=13)
        ax_acf.set_ylabel("Normalized ACF", fontsize=13)
        ax_acf.set_xlim(0, max(tau_axis_l)/12)
        ax_acf.grid(True, ls=":", alpha=0.5)
        ax_acf.legend(fontsize=10, loc="upper right")
        ax_acf.set_title("Autocorrelation Function", fontweight="bold", fontsize=14)

        if tint:
            fig.suptitle(f"Interval: {tint[0]} to {tint[1]}", fontsize=12, y=1.02)

        apply_plot_font(fig)
        plt.tight_layout()
        plt.show()

    return {
        "time": time_raw[len(time_raw) // 2],
        "Bbgd": Bbgd_vec,
        "f": f,
        "nperseg": n_actual,
        "psd_para": p_para,
        "psd_perp1": p_perp1,
        "psd_perp2": p_perp2,
        "psd_total": p_total,
        "p_total": p_total,
        "comp_ratio": comp_ratio,
        "acf_para": acf_n,
        "acf_perp": (acf_l, acf_m),
        "tau_perp1_axis": tau_axis_l,
        "tau_perp2_axis": tau_axis_m,
        "tau_para_axis": tau_axis_n,
        "f_ci": f_ci,
        "fit": fit_result,
        "slope_inertial": (
            fit_result["selected"]["segments"][0]["slope"]
            if fit_result["selected_mode"] in {"double", "triple", "quad"}
            else np.nan
        ),
        "slope_kinetic": (
            fit_result["selected"]["segments"][-1]["slope"]
            if fit_result["selected_mode"] in {"double", "triple", "quad"}
            else np.nan
        ),
        "slope_single": (
            fit_result["single"]["segments"][0]["slope"]
            if fit_result["single"] is not None
            else np.nan
        ),
        "slopes": (
            np.array([seg["slope"] for seg in fit_result["selected"]["segments"]])
            if fit_result["selected"] is not None
            else np.array([])
        ),
        "f_break": (
            fit_result["selected"]["f_breaks"][0]
            if fit_result["selected"] is not None and fit_result["selected"].get("f_breaks")
            else np.nan
        ),
        "f_breaks": (
            np.array(fit_result["selected"].get("f_breaks", []))
            if fit_result["selected"] is not None
            else np.array([])
        ),
    }


def psd_welch_sliding(
    Bwave,
    window_length,
    overlap_ratio=0.5,
    fit_range=(0.01, 5.0),
    fit_component="total",
    fit_mode="auto",
):
    time_raw = Bwave.time.data
    time_sec = _time_seconds_from_datetime64(time_raw)
    fs = 1.0 / np.median(np.diff(time_sec))

    pts_per_window = int(window_length * fs)
    step = int(pts_per_window * (1 - overlap_ratio))
    if pts_per_window <= 0 or step <= 0:
        raise ValueError("window_length and overlap_ratio produce an invalid window/step size")

    tmp_time = []
    tmp_Bbgd = []
    tmp_psd_para = []
    tmp_psd_perp = []
    tmp_psd_total = []
    tmp_slope_1 = []
    tmp_slope_2 = []
    tmp_f_break = []
    tmp_f_ci = []
    f_axis = None

    for start_idx in range(0, len(Bwave.data) - pts_per_window + 1, step):
        B_segment = Bwave.isel(time=slice(start_idx, start_idx + pts_per_window))
        out = psd_welch(
            B_segment,
            plot=False,
            fit_range=fit_range,
            fit_component=fit_component,
            fit_mode=fit_mode,
        )

        if f_axis is None:
            f_axis = out["f"]

        tmp_time.append(out["time"])
        tmp_Bbgd.append(out["Bbgd"])
        tmp_psd_para.append(out["psd_para"])
        tmp_psd_perp.append(out["psd_perp"])
        tmp_psd_total.append(out["psd_total"])
        tmp_slope_1.append(out["slope_inertial"])
        tmp_slope_2.append(out["slope_kinetic"])
        tmp_f_break.append(out["f_break"])
        tmp_f_ci.append(out["f_ci"])

    res_time = np.array(tmp_time)
    return {
        "Bbgd": ts_vec_xyz(res_time, np.array(tmp_Bbgd)),
        "psd_para": ts_spectr(res_time, f_axis, np.array(tmp_psd_para), comp_name="f"),
        "psd_perp": ts_spectr(res_time, f_axis, np.array(tmp_psd_perp), comp_name="f"),
        "psd_total": ts_spectr(res_time, f_axis, np.array(tmp_psd_total), comp_name="f"),
        "slope_inertial": np.array(tmp_slope_1),
        "slope_kinetic": np.array(tmp_slope_2),
        "f_break": np.array(tmp_f_break),
        "f_ci": np.array(tmp_f_ci),
        "time": res_time,
        "f": f_axis,
    }

if __name__ == "__main__":
    from py_space_zc import maven
    tint = ["2019-01-03T15:00:00", "2019-01-03T15:30:00"]
    B = maven.get_data(tint, "B_high")
    Bwave = B["Bmso"]
    out = psd_welch(Bwave, fit_range=(0.001, 15), fit_mode="3", )

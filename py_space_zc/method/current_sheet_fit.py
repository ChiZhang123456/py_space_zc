#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def _harris_model(t, amplitude, center_s, width_s, offset):
    width_s = np.maximum(width_s, 1e-12)
    return offset + amplitude * np.tanh((t - center_s) / width_s)


def _bifurcated_model(t, amplitude, center_s, width_s, separation_s, offset):
    width_s = np.maximum(width_s, 1e-12)
    separation_s = np.maximum(separation_s, 0.0)
    left = np.tanh((t - (center_s - separation_s)) / width_s)
    right = np.tanh((t - (center_s + separation_s)) / width_s)
    return offset + 0.5 * amplitude * (left + right)


def _get_time_and_data(ts_scalar):
    if hasattr(ts_scalar, "time"):
        time = np.asarray(ts_scalar.time.data)
    elif hasattr(ts_scalar, "coords") and "time" in ts_scalar.coords:
        time = np.asarray(ts_scalar.coords["time"].data)
    else:
        data_len = len(np.asarray(ts_scalar))
        time = np.arange(data_len, dtype=float)

    data = np.asarray(getattr(ts_scalar, "data", ts_scalar), dtype=float)
    data = np.squeeze(data)

    if data.ndim != 1:
        raise ValueError("ts_scalar must be a one dimensional scalar time series.")
    if len(time) != len(data):
        raise ValueError("time and data must have the same length.")

    return time, data


def _time_to_seconds(time):
    time = np.asarray(time)
    if np.issubdtype(time.dtype, np.datetime64):
        time_ns = time.astype("datetime64[ns]")
        t0 = time_ns[0]
        t_sec = (time_ns - t0) / np.timedelta64(1, "s")
        return np.asarray(t_sec, dtype=float), t0

    t_numeric = np.asarray(time, dtype=float)
    return t_numeric - t_numeric[0], time[0]


def _seconds_to_time(t0, seconds):
    if isinstance(t0, np.datetime64):
        ns = np.rint(np.asarray(seconds, dtype=float) * 1e9).astype("timedelta64[ns]")
        return t0 + ns
    return np.asarray(seconds, dtype=float) + t0


def _initial_guess(t, y, mode):
    duration = max(float(t[-1] - t[0]), 1e-12)
    amp0 = 0.5 * (np.nanpercentile(y, 90) - np.nanpercentile(y, 10))
    if not np.isfinite(amp0) or abs(amp0) < 1e-12:
        amp0 = np.nanstd(y)
    if y[-1] < y[0]:
        amp0 = -abs(amp0)

    offset0 = float(np.nanmedian(y))
    center0 = 0.5 * (t[0] + t[-1])
    width0 = max(duration / 10.0, 1e-6)

    if mode == "harris":
        return [amp0, center0, width0, offset0]

    return [amp0, center0, max(duration / 20.0, 1e-6), max(duration / 6.0, 0.0), offset0]


def _bounds(t, mode):
    duration = max(float(t[-1] - t[0]), 1e-12)
    dt = np.nanmedian(np.diff(t)) if len(t) > 1 else duration
    min_width = max(float(dt) * 0.25, duration / 1e5, 1e-9)

    if mode == "harris":
        return (
            [-np.inf, t[0], min_width, -np.inf],
            [np.inf, t[-1], duration * 2.0, np.inf],
        )

    return (
        [-np.inf, t[0], min_width, 0.0, -np.inf],
        [np.inf, t[-1], duration * 2.0, duration, np.inf],
    )


def _fit_metrics(y, y_fit, n_param):
    good = np.isfinite(y) & np.isfinite(y_fit)
    n_good = int(np.sum(good))
    residual = y[good] - y_fit[good]
    rss = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(rss / n_good)) if n_good else np.nan
    mae = float(np.mean(np.abs(residual))) if n_good else np.nan

    ss_tot = float(np.sum((y[good] - np.mean(y[good])) ** 2)) if n_good else np.nan
    r2 = float(1.0 - rss / ss_tot) if ss_tot and np.isfinite(ss_tot) else np.nan

    dof = n_good - n_param
    reduced_chi2 = float(rss / dof) if dof > 0 else np.nan
    aic = float(n_good * np.log(max(rss, 1e-300) / n_good) + 2 * n_param) if n_good else np.nan

    return {
        "n_points": n_good,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rss": rss,
        "chi_square": rss,
        "reduced_chi2": reduced_chi2,
        "aic": aic,
    }


def _format_seconds(value):
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _plot_parameter_text(mode, params, errors, metrics):
    lines = [
        rf"$B_0$ = {params['amplitude']:.2f} $\pm$ {errors['amplitude']:.2f}",
        rf"$T$ = {_format_seconds(params['width_s'])} s $\pm$ {_format_seconds(errors['width_s'])} s",
    ]
    if mode == "bifurcated":
        lines.append(
            rf"$\Delta t$ = {_format_seconds(params['separation_s'])} s "
            rf"$\pm$ {_format_seconds(errors['separation_s'])} s"
        )
    lines.append(rf"$R^2$ = {metrics['r2']:.3f}")
    return "\n".join(lines)


def _make_like_ts(ts_scalar, time, values, name):
    try:
        import xarray as xr

        attrs = dict(getattr(ts_scalar, "attrs", {}))
        attrs["model"] = name
        return xr.DataArray(values, coords={"time": time}, dims=["time"], attrs=attrs)
    except Exception:
        return values


def fit_current_sheet(ts_scalar, mode="harris", Plot=False, ax=None, maxfev=40000):
    """
    Fit the Bl component across a current sheet with a Harris or bifurcated model.

    Parameters
    ----------
    ts_scalar : xarray.DataArray or array_like
        Scalar time series of Bl. If a time coordinate exists, it is used for plotting
        and converted to seconds for fitting.
    mode : {"harris", "bifurcated"}, optional
        Model used for fitting. Default is "harris".
    Plot : bool, optional
        If True, plot observed Bl and fitted Bl. Default is False.
    ax : matplotlib.axes.Axes, optional
        Existing axes for plotting.
    maxfev : int, optional
        Maximum number of function evaluations for scipy.optimize.curve_fit.

    Returns
    -------
    result : dict
        Dictionary containing fitted parameters, 1 sigma parameter errors,
        covariance matrix, fitted time series, residual time series, and fit metrics.

    Notes
    -----
    Harris model:
        Bl = offset + amplitude * tanh((t - center_s) / width_s)

    Bifurcated model:
        Bl = offset + 0.5 * amplitude * [
            tanh((t - center_s + separation_s) / width_s)
            + tanh((t - center_s - separation_s) / width_s)
        ]
    """
    mode = mode.lower()
    if mode not in ("harris", "bifurcated"):
        raise ValueError("mode must be 'harris' or 'bifurcated'.")

    time_raw, data_raw = _get_time_and_data(ts_scalar)
    good = np.isfinite(data_raw)
    time = time_raw[good]
    y = data_raw[good]

    if len(y) < 5:
        raise ValueError("At least 5 finite data points are required for fitting.")
    if mode == "bifurcated" and len(y) < 6:
        raise ValueError("At least 6 finite data points are required for bifurcated fitting.")

    t, t0 = _time_to_seconds(time)
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    time = time[sort_idx]
    y = y[sort_idx]

    model = _harris_model if mode == "harris" else _bifurcated_model
    names = (
        ["amplitude", "center_s", "width_s", "offset"]
        if mode == "harris"
        else ["amplitude", "center_s", "width_s", "separation_s", "offset"]
    )

    p0 = _initial_guess(t, y, mode)
    lower, upper = _bounds(t, mode)
    popt, pcov = curve_fit(
        model,
        t,
        y,
        p0=p0,
        bounds=(lower, upper),
        maxfev=maxfev,
    )

    y_fit = model(t, *popt)
    residual = y - y_fit
    perr = np.sqrt(np.diag(pcov))

    params = {key: float(value) for key, value in zip(names, popt)}
    errors = {key: float(value) for key, value in zip(names, perr)}
    metrics = _fit_metrics(y, y_fit, len(popt))

    params["center_time"] = _seconds_to_time(t0, params["center_s"])
    if mode == "bifurcated":
        params["left_edge_time"] = _seconds_to_time(t0, params["center_s"] - params["separation_s"])
        params["right_edge_time"] = _seconds_to_time(t0, params["center_s"] + params["separation_s"])

    fit_ts = _make_like_ts(ts_scalar, time, y_fit, f"{mode}_fit")
    residual_ts = _make_like_ts(ts_scalar, time, residual, f"{mode}_residual")

    if Plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time, y, "k.", ms=4, label="Observed Bl")
        ax.plot(time, y_fit, color="tab:red", lw=2, label=f"{mode} fit")
        text = _plot_parameter_text(mode, params, errors, metrics)
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.85),
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Bl")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=False)
        ax.set_title(f"{mode.capitalize()} current sheet fit")
        plt.tight_layout()

    return {
        "mode": mode,
        "params": params,
        "param_errors": errors,
        "pcov": pcov,
        "fit": fit_ts,
        "residual": residual_ts,
        "time_s": t,
        "data": y,
        "fit_values": y_fit,
        "residual_values": residual,
        "metrics": metrics,
        "initial_guess": {key: float(value) for key, value in zip(names, p0)},
    }

import numpy as np
from scipy import signal

from py_space_zc.filt import filt
from py_space_zc.ts_vec_xyz import ts_vec_xyz
from pyrfu.pyrf import convert_fac


def _has_time_axis(value):
    return hasattr(value, "time") and hasattr(value, "data")


def _fill_nan_1d(y):
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y)
    if good.all():
        return y
    if good.sum() < 2:
        raise ValueError("Too few finite samples to interpolate missing data.")
    x = np.arange(y.size)
    return np.interp(x, x[good], y[good])


def _fill_nan_for_hilbert(data):
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        return _fill_nan_1d(data)
    return np.column_stack([_fill_nan_1d(data[:, i]) for i in range(data.shape[1])])


def _same_time_axis(a, b):
    return np.array_equal(np.asarray(a.time.data), np.asarray(b.time.data))


def _frequency_time_axis(f_min, f_max):
    f_min_has_time = _has_time_axis(f_min)
    f_max_has_time = _has_time_axis(f_max)
    if not f_min_has_time and not f_max_has_time:
        return None
    if f_min_has_time and f_max_has_time:
        if not _same_time_axis(f_min, f_max):
            raise ValueError("f_min and f_max must have the same time axis when they are time-varying.")
        return np.asarray(f_min.time.data)
    return np.asarray(f_min.time.data if f_min_has_time else f_max.time.data)


def _window_weight(n):
    if n <= 1:
        return np.ones(n)
    weight = np.hanning(n)
    if not np.any(weight > 0):
        return np.ones(n)
    return np.maximum(weight, 0.05 * np.nanmax(weight))


def _frequency_dt_seconds(freq_time):
    if freq_time.size < 2:
        raise ValueError("Time-varying frequency input needs at least two time points.")
    dt = np.nanmedian(np.diff(freq_time).astype("timedelta64[ns]").astype(float)) * 1e-9
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Frequency time axis must be strictly increasing.")
    return dt


def _hilbert_envelope_data(data):
    filled = _fill_nan_for_hilbert(data)
    env = np.abs(signal.hilbert(filled, axis=0))
    env[~np.isfinite(data)] = np.nan
    return env


def _mean_background_like(Bwave, target):
    b0_vec = np.nanmean(np.asarray(Bwave.data, dtype=float), axis=0)
    if np.any(~np.isfinite(b0_vec)):
        raise ValueError("Mean background magnetic field contains NaN or Inf.")
    if np.linalg.norm(b0_vec) == 0:
        raise ValueError("Mean background magnetic field has zero magnitude.")
    return ts_vec_xyz(target.time.data, np.tile(b0_vec, (len(target.time), 1)))


def _background_for_fac(Bwave, target, b_bgd):
    return _mean_background_like(Bwave, target) if b_bgd is None else b_bgd


def _fixed_hilbert_envelope(Bwave, f_min, f_max, order, use_fac, r_xyz, b_bgd):
    dB = filt(Bwave, f_min, f_max, order=order)
    if use_fac:
        dB = convert_fac(dB, _background_for_fac(Bwave, dB, b_bgd), r_xyz)
        dB.attrs["coordinate_system"] = "FAC"
    else:
        dB.attrs["coordinate_system"] = dB.attrs.get("coordinate_system", "input")

    envelope = _hilbert_envelope_data(dB.data)
    env = ts_vec_xyz(
        dB.time.data,
        envelope,
        attrs={
            **dB.attrs,
            "hilbert_info": "abs(hilbert(dB))",
            "coordinate_system": "FAC" if use_fac else dB.attrs.get("coordinate_system", "input"),
        },
    )
    return dB, env


def _time_varying_hilbert_envelope(Bwave, f_min, f_max, order, use_fac, r_xyz, b_bgd):
    freq_time = _frequency_time_axis(f_min, f_max)
    half_window = np.timedelta64(int(round(_frequency_dt_seconds(freq_time) * 1e9)), "ns")
    time = np.asarray(Bwave.time.data)
    n_time = time.size

    dB_sum = np.zeros_like(np.asarray(Bwave.data, dtype=float))
    env_sum = np.zeros_like(np.asarray(Bwave.data, dtype=float))
    weight_sum = np.zeros(n_time, dtype=float)

    for i, center in enumerate(freq_time):
        start = np.datetime64(center) - half_window
        stop = np.datetime64(center) + half_window
        idx = np.flatnonzero((time >= start) & (time < stop))
        if idx.size < 16:
            continue

        segment = Bwave.isel(time=idx)
        local_f_min = float(np.asarray(f_min.data)[i]) if _has_time_axis(f_min) else float(f_min)
        local_f_max = float(np.asarray(f_max.data)[i]) if _has_time_axis(f_max) else float(f_max)
        try:
            dB_segment = filt(segment, local_f_min, local_f_max, order=order)
        except ValueError:
            continue

        if use_fac:
            try:
                dB_segment = convert_fac(
                    dB_segment,
                    _background_for_fac(segment, dB_segment, b_bgd),
                    r_xyz,
                )
            except ValueError:
                continue

        env_segment = _hilbert_envelope_data(dB_segment.data)
        weight = _window_weight(idx.size)
        dB_sum[idx] += dB_segment.data * weight[:, None]
        env_sum[idx] += env_segment * weight[:, None]
        weight_sum[idx] += weight

    dB_data = np.full_like(dB_sum, np.nan, dtype=float)
    env_data = np.full_like(env_sum, np.nan, dtype=float)
    good = weight_sum > 0
    dB_data[good] = dB_sum[good] / weight_sum[good, None]
    env_data[good] = env_sum[good] / weight_sum[good, None]

    attrs = Bwave.attrs.copy()
    attrs["filter_info"] = "Time-varying Butterworth SOS with window-by-window Hilbert envelope"
    attrs["coordinate_system"] = "FAC" if use_fac else attrs.get("coordinate_system", "input")

    dB = ts_vec_xyz(Bwave.time.data, dB_data, attrs=attrs)
    env = ts_vec_xyz(
        Bwave.time.data,
        env_data,
        attrs={
            **attrs,
            "hilbert_info": "abs(hilbert(dB))",
        },
    )
    return dB, env


def hilbert_envelope(Bwave, f_min=0.0, f_max=0.0, order=4,
                     fac=True, r_xyz=None, b_bgd=None):
    """
    Bandpass magnetic field data and calculate the Hilbert envelope.

    Parameters
    ----------
    Bwave : xarray.DataArray
        Vector magnetic field time series, shape = (time, 3).
    f_min : float or time series
        Lower cutoff frequency in Hz. Time series input must have time and data.
    f_max : float or time series
        Upper cutoff frequency in Hz. Time series input must have time and data.
    order : int, optional
        Butterworth filter order. Default is 4.
    fac : bool, optional
        If True, convert each filtered window to FAC before Hilbert transform.
        If False, keep the input coordinate system. Default is True.
    r_xyz : array-like, optional
        Reference vector passed to pyrfu.pyrf.convert_fac.
    b_bgd : xarray.DataArray, optional
        Background magnetic field used to define FAC. If None and fac=True,
        the mean Bwave in each filtered interval is used. The FAC component
        order follows pyrfu.pyrf.convert_fac: [perp1, perp2, parallel].

    Returns
    -------
    dB : xarray.DataArray
        Bandpassed magnetic perturbation as ts_vec_xyz. If fac=True, components
        are [perp1, perp2, parallel].
    envelope : xarray.DataArray
        Hilbert envelope of dB as ts_vec_xyz, with the same coordinates as dB.
    """
    freq_time = _frequency_time_axis(f_min, f_max)
    if freq_time is None:
        return _fixed_hilbert_envelope(Bwave, f_min, f_max, order, fac, r_xyz, b_bgd)
    return _time_varying_hilbert_envelope(Bwave, f_min, f_max, order, fac, r_xyz, b_bgd)

import numpy as np
import xarray as xr
from scipy import signal


def _has_time_axis(value):
    return hasattr(value, "time") and hasattr(value, "data")


def _as_float_frequency(value, index=None):
    if _has_time_axis(value):
        if index is None:
            raise ValueError("index is required for time-varying frequency input.")
        return float(np.asarray(value.data)[index])
    return float(value)


def _sampling_rate(inp):
    time_diff = np.diff(inp.time.data).astype("timedelta64[ns]").astype(float)
    dt = np.nanmedian(time_diff) * 1e-9
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Input time axis must be strictly increasing.")
    return 1.0 / dt


def _fill_nan_1d(y):
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y)
    if good.all():
        return y
    if good.sum() < 2:
        raise ValueError("Too few finite samples to interpolate missing data before filtering.")
    x = np.arange(y.size)
    return np.interp(x, x[good], y[good])


def _fill_nan_for_filter(data):
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        return _fill_nan_1d(data)
    return np.column_stack([_fill_nan_1d(data[:, i]) for i in range(data.shape[1])])


def _make_sos(f_min, f_max, f_nyq, order):
    if not np.isfinite(f_min) or not np.isfinite(f_max):
        raise ValueError("Filter frequencies must be finite.")
    if f_min < 0 or f_max < 0:
        raise ValueError("Filter frequencies must be non-negative.")
    if f_min == 0 and f_max == 0:
        return None
    if f_max > f_nyq:
        raise ValueError(
            f"f_max ({f_max} Hz) exceeds the Nyquist frequency ({f_nyq:.2f} Hz). "
            f"Resolution is too low for this filter."
        )

    low = f_min / f_nyq
    high = f_max / f_nyq
    if f_min > 0 and f_max > 0:
        if not f_min < f_max:
            raise ValueError(f"f_min ({f_min} Hz) must be smaller than f_max ({f_max} Hz).")
        return signal.butter(order, [low, high], btype="bandpass", output="sos")
    if f_min > 0:
        return signal.butter(order, low, btype="highpass", output="sos")
    return signal.butter(order, high, btype="lowpass", output="sos")


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


def _filter_fixed(inp, f_min, f_max, fs, f_nyq, order):
    sos = _make_sos(float(f_min), float(f_max), f_nyq, order)
    if sos is None:
        return inp
    filtered_data = signal.sosfiltfilt(sos, _fill_nan_for_filter(inp.data), axis=0)
    out = xr.DataArray(
        filtered_data,
        coords=inp.coords,
        dims=inp.dims,
        attrs=inp.attrs.copy()
    )
    out.attrs["filter_info"] = f"Butterworth SOS {f_min}-{f_max} Hz"
    out.attrs["sampling_rate"] = f"{fs:.2f} Hz"
    return out


def _filter_time_varying(inp, f_min, f_max, fs, f_nyq, order):
    freq_time = _frequency_time_axis(f_min, f_max)
    if freq_time is None:
        raise ValueError("Internal error: missing frequency time axis.")
    if freq_time.size < 2:
        raise ValueError("Time-varying frequency input needs at least two time points.")

    freq_dt = np.nanmedian(np.diff(freq_time).astype("timedelta64[ns]").astype(float)) * 1e-9
    if not np.isfinite(freq_dt) or freq_dt <= 0:
        raise ValueError("Frequency time axis must be strictly increasing.")

    data = np.asarray(inp.data)
    out_shape = data.shape
    sum_data = np.zeros(out_shape, dtype=float)
    sum_weight = np.zeros(out_shape[0], dtype=float)
    time = np.asarray(inp.time.data)
    half_window = np.timedelta64(int(round(freq_dt * 1e9)), "ns")

    for i, center in enumerate(freq_time):
        local_f_min = _as_float_frequency(f_min, i)
        local_f_max = _as_float_frequency(f_max, i)
        if local_f_min == 0 and local_f_max == 0:
            continue
        sos = _make_sos(local_f_min, local_f_max, f_nyq, order)
        start = np.datetime64(center) - half_window
        stop = np.datetime64(center) + half_window
        mask = (time >= start) & (time < stop)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        padlen = 3 * (2 * len(sos) + 1)
        if idx.size <= padlen:
            continue

        segment = _fill_nan_for_filter(data[idx])
        filtered_segment = signal.sosfiltfilt(sos, segment, axis=0)
        weight = _window_weight(idx.size)
        sum_weight[idx] += weight
        if data.ndim == 1:
            sum_data[idx] += filtered_segment * weight
        else:
            sum_data[idx] += filtered_segment * weight[:, None]

    filtered_data = np.full(out_shape, np.nan, dtype=float)
    good = sum_weight > 0
    if data.ndim == 1:
        filtered_data[good] = sum_data[good] / sum_weight[good]
    else:
        filtered_data[good] = sum_data[good] / sum_weight[good, None]

    out = xr.DataArray(
        filtered_data,
        coords=inp.coords,
        dims=inp.dims,
        attrs=inp.attrs.copy()
    )
    out.attrs["filter_info"] = (
        f"Time-varying Butterworth SOS, window half-width = {freq_dt:.3f} s"
    )
    out.attrs["sampling_rate"] = f"{fs:.2f} Hz"
    return out


def filt(inp, f_min: float = 0.0, f_max: float = 0.0, order: int = 4):
    """
    Apply a stable digital filter using Second-Order Sections (SOS).
    Supports Bandpass, Highpass, and Lowpass filters.
    f_min and f_max can be either fixed numbers or time series such as ts_scalar.
    For time-varying frequencies, each frequency time marks a local filter window
    from t - dt to t + dt, where dt is the median frequency-grid cadence.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series data. Must contain a 'time' coordinate.
    f_min : float, optional
        Lower cutoff frequency (Hz). If 0, performs a Lowpass filter.
        Can also be a time series with matching time coordinates.
    f_max : float, optional
        Upper cutoff frequency (Hz). If 0, performs a Highpass filter.
        Can also be a time series with matching time coordinates.
    order : int, optional
        Filter order. Default is 4 (equivalent to 8 after filtfilt).

    Returns
    -------
    out : xarray.DataArray
        Filtered time series with zero phase shift.
    """

    fs = _sampling_rate(inp)
    f_nyq = fs / 2.0

    if _frequency_time_axis(f_min, f_max) is not None:
        return _filter_time_varying(inp, f_min, f_max, fs, f_nyq, order)

    return _filter_fixed(inp, f_min, f_max, fs, f_nyq, order)

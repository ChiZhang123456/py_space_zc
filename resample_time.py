#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import bisect
import logging

# 3rd party imports
import numpy as np
import xarray as xr
from scipy import interpolate


logging.captureWarnings(True)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def _guess_sampling_frequency(ref_time):
    r"""Compute sampling frequency of the time line."""

    n_data = len(ref_time)

    sfy1 = 1 / (ref_time[1] - ref_time[0])
    sfy = None
    not_found = True

    if n_data == 2:
        sfy = sfy1
        not_found = False

    cur, max_try = [2, 10]

    while not_found and cur < n_data and cur - 3 < max_try:
        sfy = 1 / (ref_time[cur] - ref_time[cur - 1])

        if np.absolute(sfy - sfy1) < sfy * 0.001:
            not_found = False

            sfy = (sfy + sfy1) / 2
            break

        sfy = sfy1
        cur += 1

    if not_found:
        raise RuntimeError(f"Cannot guess sampling frequency. Tried {max_try:d} times")

    return sfy


def _average(inp_time, inp_data, ref_time, thresh, dt2):
    r"""Resample inp_data to timeline of ref_time, using half-window of dt2.
    Points above std*tresh are excluded. thresh=0 turns off this option.
    """

    try:
        out_data = np.zeros([len(ref_time), *inp_data.shape[1:]])
    except IndexError:
        inp_data = inp_data[:, None]
        out_data = np.zeros((len(ref_time), inp_data.shape[1]))

    for i, ref_t in enumerate(ref_time):
        idx_l = bisect.bisect_left(inp_time, ref_t - dt2)
        idx_r = bisect.bisect_right(inp_time, ref_t + dt2)

        idx = np.arange(idx_l, idx_r)

        if idx.size == 0:
            out_data[i, ...] = np.nan
        else:
            if thresh:
                std_ = np.std(inp_data[idx, ...], axis=0)
                mean_ = np.mean(inp_data[idx, ...], axis=0)

                assert any(np.isnan(std_))

                for j, stdd in enumerate(std_):
                    if not np.isnan(stdd):
                        idx_r = bisect.bisect_right(
                            inp_data[idx, j + 1] - mean_[j],
                            thresh * stdd,
                        )
                        if idx_r:
                            out_data[i, j + 1] = np.mean(
                                inp_data[idx[idx_r], j + 1],
                                axis=0,
                            )
                        else:
                            out_data[i, j + 1] = np.nan
                    else:
                        out_data[i, ...] = np.nan

            else:
                out_data[i, ...] = np.mean(inp_data[idx, ...], axis=0)

    if out_data.ndim > 1 and out_data.shape[1] == 1:
        out_data = out_data[:, 0]

    return out_data

def resample_time(time, data, ref_time,
                   method: str = "",
                   f_s: float = None,
                   window: int = None,
                   thresh: float = 0,):

    options = {"method": method, "f_s": f_s, "window": window, "thresh": thresh}
    flag_do = "check"
    if method:
        flag_do = "interpolation"
    if f_s is not None:
        sfy = f_s
    elif window is not None:
        sfy = 1 / window
    else:
        sfy = None

    inp_time = time.view("i8") * 1e-9
    ref_time = ref_time.view("i8") * 1e-9

    if flag_do == "check":
        if len(ref_time) > 1:
            if not sfy:
                sfy = _guess_sampling_frequency(ref_time)

            if len(inp_time) / (inp_time[-1] - inp_time[0]) > 2 * sfy:
                flag_do = "average"
                logging.info("Using averages in resample")
            else:
                flag_do = "interpolation"
        else:
            flag_do = "interpolation"

    assert flag_do in ["average", "interpolation"]

    if flag_do == "average":
        assert not method, "cannot mix interpolation and averaging flags"

        if not sfy:
            sfy = _guess_sampling_frequency(ref_time)

        out_data = _average(inp_time, data, ref_time, thresh, 0.5 / sfy)

    else:
        if not method:
            method = "linear"

        # If time series agree, no interpolation is necessary.
        if len(inp_time) == len(ref_time) and all(inp_time == ref_time):
            out_data = data.copy()
            return out_data

        tck = interpolate.interp1d(
            inp_time,
            data,
            kind=method,
            axis=0,
            fill_value="extrapolate",
        )
        out_data = tck(ref_time)
    return out_data

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot MAVEN IUVS L1C periapse altitude profiles.
"""

import math
import re
from pathlib import Path

import numpy as np

from .read_l1c_per import read_l1c_per


def _default_grid(n_scan):
    ncol = min(4, max(1, math.ceil(math.sqrt(n_scan))))
    nrow = math.ceil(n_scan / ncol)
    return nrow, ncol


def _format_iuvs_time(value, include_date=True):
    text = str(value).replace("UTC", "").strip()
    months = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    match = re.search(
        r"(?P<year>\d{4})/\d+\s+(?P<mon>[A-Za-z]{3})\s+"
        r"(?P<day>\d{1,2})\s+(?P<hour>\d{2}):(?P<minute>\d{2})",
        text,
    )
    if match:
        parts = match.groupdict()
        date = f"{parts['year']}-{months.get(parts['mon'], parts['mon'])}-{int(parts['day']):02d}"
        hm = f"{parts['hour']}:{parts['minute']}"
        return f"{date} {hm}" if include_date else hm

    if "T" in text:
        date, rest = text.split("T", 1)
        hm = rest[:5]
        return f"{date} {hm}" if include_date else hm
    return text


def _time_interval_title(data):
    start = data.get("time_start_utc", "")
    stop = data.get("time_stop_utc", "")
    if not start or not stop:
        return Path(data["filename"]).name

    start_text = _format_iuvs_time(start, include_date=True)
    stop_text = _format_iuvs_time(stop, include_date=False)
    if start_text[:10] and _format_iuvs_time(stop, include_date=True).startswith(start_text[:10]):
        return f"{start_text}-{stop_text} UTC"
    return f"{start_text}-{_format_iuvs_time(stop, include_date=True)} UTC"


def _orbit_title_prefix(data):
    orbit = data.get("orbit_number", None)
    if orbit is None:
        match = re.search(r"orbit(\d+)", Path(data["filename"]).name)
        if match:
            orbit = int(match.group(1))
    if orbit is None:
        return ""
    try:
        orbit = int(orbit)
    except (TypeError, ValueError):
        pass
    return f"Orbit {orbit} "


def plot_l1c_per(
    filename,
    emission="H lyman-Alpha",
    altitude_range=None,
    sza_altitude_range=(110.0, 150.0),
    xlog=False,
    figsize=None,
    save_path=None,
    show=True,
):
    """
    Plot all scan altitude profiles from one L1C periapse file on one axis.

    Parameters
    ----------
    filename : str or pathlib.Path
        Local ``*.fits`` or ``*.fits.gz`` L1C periapse file.
    emission : str, optional
        Emission name. Default is H Ly-alpha.
    altitude_range : tuple or None, optional
        Optional y-axis altitude range in km.
    sza_altitude_range : tuple or None, optional
        Altitude range used to label one representative SZA per scan.
    xlog : bool, optional
        Use a logarithmic radiance axis.
    figsize : tuple or None, optional
        Matplotlib figure size.
    save_path : str or pathlib.Path or None, optional
        Save figure if provided.
    show : bool, optional
        Call ``plt.show()`` before returning.

    Returns
    -------
    tuple
        ``(fig, ax, data)`` where data is the dictionary from
        :func:`read_l1c_per`.
    """
    import matplotlib.pyplot as plt

    data = read_l1c_per(filename, emission=emission, sza_altitude_range=sza_altitude_range)
    altitude = data["altitude"]
    brightness = data["brightness"]
    sza_profile = data["sza_profile"]

    n_scan = brightness.shape[0]
    if figsize is None:
        figsize = (6.8, 5.2)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, n_scan))

    for i in range(n_scan):
        alt_i = altitude[i]
        rad_i = brightness[i]
        good = np.isfinite(alt_i) & np.isfinite(rad_i)
        if xlog:
            good &= rad_i > 0
        if altitude_range is not None:
            good &= (alt_i >= altitude_range[0]) & (alt_i <= altitude_range[1])

        label = f"Scan {i + 1}"
        if np.isfinite(sza_profile[i]):
            label = f"SZA {sza_profile[i]:.1f} deg"
        ax.plot(rad_i[good], alt_i[good], color=colors[i], lw=1.5, label=label)

    if xlog:
        ax.set_xscale("log")

    if altitude_range is not None:
        ax.set_ylim(altitude_range)

    ax.set_xlabel(f"{data['emission']} radiance ({data['radiance_unit']})")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{_orbit_title_prefix(data)}{_time_interval_title(data)}")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Scan SZA", fontsize=8, title_fontsize=9, loc="best")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()

    return fig, ax, data

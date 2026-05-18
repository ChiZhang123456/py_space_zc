#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Find and read local MAVEN IUVS L1C periapse files.
"""

from datetime import datetime
from pathlib import Path
import re

from .plot_l1c_per import plot_l1c_per as _plot_l1c_per
from .read_l1c_per import read_l1c_per
from ..get_base_path import get_base_path

DEFAULT_L1C_PER_ROOT = Path(get_base_path()) / "iuvs" / "l1c" / "periapse"


def _parse_filename(path):
    name = Path(path).name
    match = re.search(r"orbit(?P<orbit>\d+)_(?P<time>\d{8}T\d{6})", name)
    if not match:
        return None

    version_match = re.search(r"_v(?P<version>\d+)_r(?P<revision>\d+)", name)
    version = int(version_match.group("version")) if version_match else -1
    revision = int(version_match.group("revision")) if version_match else -1

    return {
        "path": Path(path),
        "orbit": int(match.group("orbit")),
        "time": datetime.strptime(match.group("time"), "%Y%m%dT%H%M%S"),
        "version": version,
        "revision": revision,
    }


def _parse_input_time(value):
    if isinstance(value, datetime):
        return value

    text = str(value).strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(
        "time must look like '2017-01-02T00:01:00', "
        "'2017-01-02 00:01:00', or '20170102T000100'"
    )


def _iter_l1c_per_files(data_root):
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"IUVS L1C periapse root not found: {root}")
    yield from root.rglob("mvn_iuv_l1c_periapse-orbit*.fits.gz")


def find_l1c_per_file(time=None, orbit=None, data_root=DEFAULT_L1C_PER_ROOT):
    """
    Find a local IUVS L1C periapse file by orbit or nearest file time.

    Parameters
    ----------
    time : str or datetime, optional
        Target UTC time. The nearest file timestamp in the filename is used.
    orbit : int, optional
        Orbit number. Exact orbit match is used.
    data_root : str or pathlib.Path, optional
        Root directory containing ``l1c/periapse/YYYY/MM`` files.

    Returns
    -------
    pathlib.Path
        Best matching local FITS file.
    """
    records = []
    for path in _iter_l1c_per_files(data_root):
        record = _parse_filename(path)
        if record is not None:
            records.append(record)

    if not records:
        raise FileNotFoundError(f"No L1C periapse FITS files found under {data_root}")

    if orbit is not None:
        orbit = int(orbit)
        matches = [record for record in records if record["orbit"] == orbit]
        if not matches:
            raise FileNotFoundError(f"No local L1C periapse FITS file found for orbit {orbit}")
        best = max(matches, key=lambda item: (item["version"], item["revision"], item["time"]))
        return best["path"]

    if time is None:
        raise ValueError("Provide either time or orbit")

    target_time = _parse_input_time(time)
    best = min(
        records,
        key=lambda item: (
            abs((item["time"] - target_time).total_seconds()),
            -item["version"],
            -item["revision"],
        ),
    )
    return best["path"]


def get_l1c_per(
    time=None,
    orbit=None,
    plot=False,
    data_root=DEFAULT_L1C_PER_ROOT,
    emission="H lyman-Alpha",
    sza_altitude_range=(110.0, 150.0),
    **plot_kwargs,
):
    """
    Find a local L1C periapse file, read it, and optionally plot it.

    Examples
    --------
    ``data = get_l1c_per(time='2017-01-02T00:01:00')``

    ``data = get_l1c_per(orbit=5717, plot=True)``
    """
    filename = find_l1c_per_file(time=time, orbit=orbit, data_root=data_root)
    data = read_l1c_per(
        filename,
        emission=emission,
        sza_altitude_range=sza_altitude_range,
    )

    if time is not None and orbit is None:
        target_time = _parse_input_time(time)
        file_record = _parse_filename(filename)
        data["target_time_utc"] = target_time.strftime("%Y-%m-%dT%H:%M:%S")
        data["file_time_difference_seconds"] = abs(
            (file_record["time"] - target_time).total_seconds()
        )

    if plot:
        _plot_l1c_per(
            filename,
            emission=emission,
            sza_altitude_range=sza_altitude_range,
            **plot_kwargs,
        )

    return data

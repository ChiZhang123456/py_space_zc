#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read MAVEN IUVS L1C periapse altitude profiles.
"""

from pathlib import Path
import re

import numpy as np
from astropy.io import fits


FILL_LIMIT = -1.0e20


def _decode(value):
    """Return a stripped string from FITS bytes or scalar values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _clean_float_array(values):
    arr = np.asarray(values, dtype=float).copy()
    arr[(arr <= FILL_LIMIT) | ~np.isfinite(arr)] = np.nan
    return arr


def _get_column(data, name, default=None):
    names = set(data.columns.names)
    if name in names:
        return data[name]
    return default


def _scan_time_bounds(integration_data):
    if integration_data is None or "UTC" not in integration_data.columns.names:
        empty = np.array([], dtype=object)
        return empty, empty, empty

    utc = np.asarray(integration_data["UTC"])
    if utc.ndim == 1:
        utc = utc[:, np.newaxis]

    start_times = []
    mid_times = []
    stop_times = []
    for row in utc:
        vals = [_decode(v) for v in np.ravel(row)]
        vals = [v for v in vals if v and v.upper() not in {"N/A", "NULL", "NONE"}]
        if vals:
            start_times.append(vals[0])
            mid_times.append(vals[len(vals) // 2])
            stop_times.append(vals[-1])
        else:
            start_times.append("")
            mid_times.append("")
            stop_times.append("")
    return (
        np.asarray(start_times, dtype=object),
        np.asarray(mid_times, dtype=object),
        np.asarray(stop_times, dtype=object),
    )


def _profile_sza(altitude, sza, altitude_range=(110.0, 150.0)):
    out = np.full(sza.shape[0], np.nan)
    for i in range(sza.shape[0]):
        good = np.isfinite(sza[i]) & np.isfinite(altitude[i])
        if altitude_range is not None:
            good &= (altitude[i] >= altitude_range[0]) & (altitude[i] <= altitude_range[1])
        if not np.any(good):
            good = np.isfinite(sza[i])
        if np.any(good):
            out[i] = np.nanmedian(sza[i, good])
    return out


def _read_observation_metadata(observation_data):
    if observation_data is None or len(observation_data) == 0:
        return {}

    row = observation_data[0]
    meta = {}
    for key in ["ORBIT_NUMBER", "SOLAR_LONGITUDE", "OBSERVATION_ID", "PRODUCT_ID"]:
        if key in observation_data.columns.names:
            value = row[key]
            if isinstance(value, bytes):
                value = _decode(value)
            elif np.isscalar(value):
                value = value.item() if hasattr(value, "item") else value
            meta[key.lower()] = value
    return meta


def _metadata_from_filename(filename):
    name = Path(filename).name
    meta = {}

    orbit_match = re.search(r"orbit(\d+)", name)
    if orbit_match:
        meta["orbit_number"] = int(orbit_match.group(1))

    time_match = re.search(r"_(\d{8}T\d{6})", name)
    if time_match:
        meta["file_time_utc"] = time_match.group(1)

    return meta


def read_l1c_per(filename, emission="H lyman-Alpha", sza_altitude_range=(110.0, 150.0)):
    """
    Read altitude-binned profiles from a MAVEN IUVS L1C periapse FITS file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Local ``*.fits`` or ``*.fits.gz`` L1C periapse file.
    emission : str, optional
        Emission name in the ``emission_features`` HDU. The default reads
        H Ly-alpha.
    sza_altitude_range : tuple or None, optional
        Altitude range used to compute one representative SZA per scan.

    Returns
    -------
    dict
        Dictionary with scan by altitude arrays. Important keys are
        ``altitude`` in km, ``sza`` in degrees, ``brightness`` and ``radiance``.
    """
    filename = Path(filename)

    with fits.open(filename, memmap=False) as hdul:
        emission_names = [_decode(v) for v in hdul["emission_features"].data["EMISSION"]]
        lower_names = [name.lower() for name in emission_names]
        try:
            emission_index = lower_names.index(emission.lower())
        except ValueError as exc:
            available = ", ".join(emission_names)
            raise ValueError(f"Emission {emission!r} not found. Available: {available}") from exc

        radiance_data = hdul["radiance_altbin"].data
        geometry_data = hdul["geometry_altbin"].data
        integration_data = hdul["integration"].data if "integration" in hdul else None
        observation_data = hdul["observation"].data if "observation" in hdul else None

        radiance = _clean_float_array(radiance_data["PROFILE"][:, :, emission_index])
        altitude = _clean_float_array(geometry_data["TANGENT_ALT"])
        sza = _clean_float_array(geometry_data["SZA"])

        scan_start_utc, scan_mid_utc, scan_stop_utc = _scan_time_bounds(integration_data)

        out = {
            "filename": str(filename),
            "emission": emission_names[emission_index],
            "emission_index": emission_index,
            "time_utc": scan_mid_utc,
            "scan_start_utc": scan_start_utc,
            "scan_stop_utc": scan_stop_utc,
            "time_start_utc": scan_start_utc[0] if len(scan_start_utc) else "",
            "time_stop_utc": scan_stop_utc[-1] if len(scan_stop_utc) else "",
            "altitude": altitude,
            "sza": sza,
            "sza_profile": _profile_sza(altitude, sza, sza_altitude_range),
            "brightness": radiance,
            "radiance": radiance,
            "radiance_unit": "kR",
        }

        for key in ["LAT", "LON", "LOCAL_TIME"]:
            value = _get_column(geometry_data, key)
            if value is not None:
                out[key.lower()] = _clean_float_array(value)

        out.update(_metadata_from_filename(filename))
        out.update(_read_observation_metadata(observation_data))

    return out

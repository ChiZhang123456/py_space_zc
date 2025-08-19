# -*- coding: utf-8 -*-
"""
Tianwen-1 Data Retrieval Module

Author: Chi Zhang (zc199508@bu.edu)
Last updated: 2024

This module provides functionality to extract and structure Tianwen-1 spacecraft data
within a specified time interval and for a specific instrument or variable.

Supported variables include:
- "B": 1 Hz magnetic field and spacecraft position data (from MOMAG).
- "minpa_mod1": Mode 1 data from the MINPA instrument, including full 5D DEF.

Dependencies:
- SciPy (for loading .mat files)
- h5py (if future expansion to HDF5 is needed)
- xarray
- py_space_zc internal utilities
"""

import os
import numpy as np
import datetime
import glob
import scipy.io as sio
import xarray as xr
import h5py

from py_space_zc.tianwen_1 import get_base_path
from py_space_zc import (
    irf_time, get_cdf_var, year_month_day, tint_data,
    ts_scalar, ts_vec_xyz, ts_spectr, ts_skymap, read_time_from_file, loadmat, norm, normalize
)
from py_space_zc.vdf import create_pdist_skymap


# =============================================================================
# Main data retrieval function
# =============================================================================

def get_data(tint, var):
    """
    Retrieve Tianwen-1 data for the specified time interval and variable.

    Parameters
    ----------
    tint : list of str
        Time interval in ISO 8601 format. Example:
        ["2018-11-01T01:00:00", "2018-11-01T02:00:00"]

    var : str
        Variable to retrieve. Supported values:
        - "B" : Magnetic field and position data (1Hz, in MSO coordinates)
        - "minpa_mod1" : 5D differential energy flux (DEF) data from MINPA mode 1

    Returns
    -------
    res : dict
        A dictionary containing requested data. Keys depend on `var`:

        For var == "B":
            {
                "Bmso": TSeries of magnetic field [nT] (xarray.DataArray),
                "Pmso": TSeries of spacecraft position [km] (xarray.DataArray)
            }

        For var == "minpa_mod1":
            {
                "time": np.ndarray of datetime64,
                "energy": ndarray of shape (ntime, nenergy),
                "DEF": ndarray of shape (ntime, nenergy, nphi, ntheta, nmass),
                "phi": 1D array of azimuth angles,
                "theta": 1D array of elevation angles,
                "mass": 1D array of mass channels
            }

    Example
    -------
    >>> tint = ["2018-11-01T01:00:00", "2018-11-01T02:00:00"]
    >>> res = get_data(tint, "B")
    >>> res["Bmso"].plot()
    """

    # Normalize and parse time interval
    start_time, end_time = np.datetime64(tint[0]), np.datetime64(tint[1])

    # Base path to MAVEN data (defined in config)
    base_path = get_base_path()

    # Define data paths and filename patterns
    data_config = {
        "B": {
            "path": os.path.join(base_path, "MAG", "matlab_data"),
            "filename_format": "{date}_1s.mat",
        },
        "minpa_mod1": {
            "path": os.path.join(base_path, "MINPA_DATA", "DATA_MINPA"),
            "filename_format": "????????_??????_????????_??????_mod01.mat",
        },
    }

    config = data_config.get(var)
    if not config:
        raise ValueError(f"Unsupported variable: {var}")

    # Generate list of days to search
    dates_to_read = np.arange(
        start_time.astype("datetime64[D]"),
        end_time.astype("datetime64[D]") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )

    # =========================================================================
    # CASE 1: MAGNETIC FIELD (1 Hz) in MSO coordinates
    # =========================================================================
    if var == "B":
        res = {
            "time": np.array([], dtype="datetime64[ns]"),
            "Bmso": np.empty((0, 3), dtype=float),
            "Pmso": np.empty((0, 3), dtype=float),
        }

        for date_file in dates_to_read:
            filename = os.path.join(
                config["path"],
                config["filename_format"].format(date=date_file.astype(object).strftime("%Y%m%d"))
            )

            if os.path.exists(filename):
                mat_data = loadmat(filename)

                # Convert MATLAB datenum to datetime64
                time_array = np.squeeze(irf_time(mat_data["time"], "datenum>datetime64"))

                # Time mask within requested interval
                mask = (time_array >= start_time) & (time_array <= end_time)

                # Append masked data
                res["time"] = np.concatenate((res["time"], time_array[mask]))
                res["Bmso"] = np.vstack((res["Bmso"], mat_data["Bmso"][mask, :]))
                res["Pmso"] = np.vstack((res["Pmso"], mat_data["Pmso"][mask, :]))

        # Convert to TSeries (xarray.DataArray with attributes)
        B = ts_vec_xyz(
            res["time"], res["Bmso"],
            attrs={
                "name": "Magnetic field",
                "instrument": "MOMAG",
                "UNITS": "nT",
                "coordinates": "MSO",
                "resolution": "1Hz"
            }
        )
        if np.max(norm(res["Pmso"])) > 200000:
            res["Pmso"] /= 3390

        if np.min(norm(res["Pmso"])) < 20:
            res["Pmso"] *= 3390

        P = ts_vec_xyz(
            res["time"], res["Pmso"],
            attrs={
                "name": "Spacecraft position",
                "UNITS": "km",
                "coordinates": "MSO",
                "resolution": "1Hz"
            }
        )

        return {"Bmso": B, "Pmso": P}

    # =========================================================================
    # CASE 2: MINPA MODE 1 (5D DEF: energy, phi, theta, mass, time)
    # =========================================================================
    elif var == "minpa_mod1":
        nenergy, nphi, ntheta, nmass = 64, 16, 4, 8
        res = {
            "time": np.array([], dtype="datetime64[ns]"),
            "energy": np.empty((0, nenergy)),
            "DEF": np.empty((0, nenergy, nphi, ntheta, nmass)),
            "phi": np.array([], dtype=np.float64),
            "theta": np.array([], dtype=np.float64),
            "mass": np.array([], dtype=np.float64),
        }

        file_pattern = os.path.join(config["path"], config["filename_format"])
        filelist = glob.glob(file_pattern)

        for filename in filelist:
            file_start, file_end = read_time_from_file(filename)

            # Only load file if it overlaps the requested time interval
            if file_start <= end_time and file_end >= start_time:
                mat_data = loadmat(filename)

                time_array = irf_time(mat_data["time"], "datenum>datetime64").squeeze()
                mask = (time_array >= start_time) & (time_array <= end_time)

                res["time"] = np.concatenate((res["time"], time_array[mask]))
                res["energy"] = np.vstack((res["energy"], mat_data["energy"][mask, :]))
                res["DEF"] = np.vstack((res["DEF"], mat_data["DEF"][mask, :, :, :, :]))
                res["phi"] = mat_data["phi"]
                res["theta"] = mat_data["theta"]
                res["mass"] = mat_data["mass"]

        return res


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    tint = ["2022-10-18T13:08:00", "2022-10-18T13:13:00"]

    # Example: Retrieve magnetic field
    # B = get_data(tint, "B")
    # print(B["Bmso"].data.shape)

    # Example: Retrieve MINPA Mode 1 DEF
    minpa = get_data(tint, "minpa_mod1")
    print(minpa["DEF"].shape)

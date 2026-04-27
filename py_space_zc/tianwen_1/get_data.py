"""
Tianwen-1 Data Retrieval Module
===============================
Author: Chi Zhang

This module provides a unified interface to retrieve selected Tianwen-1
science data products within a user-specified time interval.

Currently supported data products include:
1. "B"
   Low-resolution (1 Hz) magnetic field and spacecraft position data
   from the MOMAG instrument in MSO coordinates.

2. "B_high"
   High-resolution (32 Hz) magnetic field and spacecraft position data
   from the MOMAG instrument in MSO coordinates.

3. "minpa_mod1"
   Mode 1 ion data from the MINPA instrument, including time-dependent
   differential energy flux (DEF) and corresponding energy, phi, theta,
   and mass dimensions.

Notes:
For MINPA Mode 1:
   - Returned DEF has shape [ntime, nenergy, nphi, ntheta, nmass]
   - Returned energy has shape [ntime, nenergy]

Example
-------
>>> tint = ["2025-06-15T11:57:00", "2025-06-15T12:10:00"]
>>> out = get_data(tint, "B_high")
>>> print(out["Bmso"].data.shape)

>>> minpa = get_data(tint, "minpa_mod1")
>>> print(minpa["DEF"].shape)
"""

import os
import glob
import numpy as np
import pandas as pd
from py_space_zc.tianwen_1 import get_base_path
from py_space_zc import (
    irf_time, year_month_day, ts_vec_xyz,
    read_time_from_file, loadmat, norm
)


def read_mag_dat(filename):
    """
    Read Tianwen-1 MOMAG .dat file.

    Parameters:
        filename (str): Path to the .dat file.

    Returns:
        time (np.ndarray): 1D array of datetime64[ns].
        Bmso (np.ndarray): Nx3 array of Magnetic Field components (nT).
        Pmso (np.ndarray): Nx3 array of Spacecraft Position (km).
    """
    # 1. Load data using pandas
    # comment='#' ignores metadata lines starting with ##
    # sep='\s+' handles multiple spaces between columns
    df = pd.read_csv(filename,comment='#',sep='\s+',header=None,usecols=[0, 2, 3, 4, 5, 6, 7],
                     names=['time', 'Bx', 'By', 'Bz', 'Px', 'Py', 'Pz'],
                     engine='python')

    # 2. Convert ISO 8601 strings to numpy datetime64[ns]
    # .values returns the underlying numpy array
    time = pd.to_datetime(df['time']).values.astype('datetime64[ns]')

    # 3. Extract MSO coordinates as Nx3 numpy arrays
    Bmso = df[['Bx', 'By', 'Bz']].values
    Pmso = df[['Px', 'Py', 'Pz']].values

    return time, Bmso, Pmso


#%%
def get_data(tint, var):
    """
    Retrieve Tianwen-1 data for the specified time interval and variable.
    """

    start_time = np.datetime64(tint[0])
    end_time = np.datetime64(tint[1])

    base_path = get_base_path()

    data_config = {
        "B": {
            "path": os.path.join(base_path, "MAG"),
            "filename_format": "TW1_MOMAG_MSO_01Hz_{date}_*.dat",
            "resolution": "1Hz",
        },
        "B_high": {
            "path": os.path.join(base_path, "MAG"),
            "filename_format": "TW1_MOMAG_MSO_32Hz_{date}_*.dat",
            "resolution": "32Hz",
        },
        "minpa_mod1": {
            "path": os.path.join(base_path, "MINPA_DATA", "DATA_MINPA"),
            "filename_format": "????????_??????_????????_??????_mod01.mat",
        },
    }

    config = data_config.get(var)
    if config is None:
        raise ValueError(f"Unsupported variable: {var}")

    dates_to_read = np.arange(
        start_time.astype("datetime64[D]"),
        end_time.astype("datetime64[D]") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )

#%%
    # =========================================================================
    # CASE 1: MAGNETIC FIELD (1 Hz / 32 Hz)
    # =========================================================================
    if var == "B":
        res = {
            "time": np.array([], dtype="datetime64[ns]"),
            "Bmso": np.empty((0, 3), dtype=float),
            "Pmso": np.empty((0, 3), dtype=float),
        }

        for date_file in dates_to_read:
            year, month, day = year_month_day(date_file)
            date_str = date_file.astype(object).strftime("%Y%m%d")
            pattern = os.path.join(config["path"],year,config["filename_format"].format(date=date_str))

            filelist = sorted(glob.glob(pattern))

            if len(filelist) == 0:
                continue

            for filename in filelist:
                try:
                    time_array, Bmso, Pmso = read_mag_dat(filename)

                    time_array = np.asarray(time_array)
                    Bmso = np.asarray(Bmso)
                    Pmso = np.asarray(Pmso)

                    mask = (time_array >= start_time) & (time_array <= end_time)

                    if np.any(mask):
                        res["time"] = np.concatenate((res["time"], time_array[mask]))
                        res["Bmso"] = np.vstack((res["Bmso"], Bmso[mask, :]))
                        res["Pmso"] = np.vstack((res["Pmso"], Pmso[mask, :]))

                except Exception as e:
                    print(f"Warning: failed to read {filename}: {e}")

        if res["time"].size == 0:
            return {"Bmso": None, "Pmso": None}

        sort_idx = np.argsort(res["time"])
        res["time"] = res["time"][sort_idx]
        res["Bmso"] = res["Bmso"][sort_idx, :]
        res["Pmso"] = res["Pmso"][sort_idx, :]

        B = ts_vec_xyz(
            res["time"],
            res["Bmso"],
            attrs={
                "name": "Magnetic field",
                "instrument": "MOMAG",
                "UNITS": "nT",
                "coordinates": "MSO",
                "resolution": config["resolution"],
            }
        )

        # Attempt to correct spacecraft position units automatically.
        if res["Pmso"].shape[0] > 0:
            if np.nanmax(norm(res["Pmso"])) > 200000:
                res["Pmso"] = res["Pmso"] / 3390.0

            if np.nanmin(norm(res["Pmso"])) < 20:
                res["Pmso"] = res["Pmso"] * 3390.0

        P = ts_vec_xyz(
            res["time"],
            res["Pmso"],
            attrs={
                "name": "Spacecraft position",
                "UNITS": "km",
                "coordinates": "MSO",
                "resolution": config["resolution"],
            }
        )

        return {"Bmso": B, "Pmso": P}
#%%
    elif var == "B_high":
        res = {
            "time": np.array([], dtype="datetime64[ns]"),
            "Bmso": np.empty((0, 3), dtype=float),
            "Pmso": np.empty((0, 3), dtype=float),
        }
        for date_file in dates_to_read:
            year, month, day = year_month_day(date_file)
            date_str = date_file.astype(object).strftime("%Y%m%d")
            pattern = os.path.join(config["path"],year,config["filename_format"].format(date=date_str))
            filelist = sorted(glob.glob(pattern))
            if len(filelist) == 0:
                continue
            for filename in filelist:
                try:
                    time_array, Bmso, Pmso = read_mag_dat(filename)
                    time_array = np.asarray(time_array)
                    Bmso = np.asarray(Bmso)
                    Pmso = np.asarray(Pmso)
                    mask = (time_array >= start_time) & (time_array <= end_time)
                    if np.any(mask):
                        res["time"] = np.concatenate((res["time"], time_array[mask]))
                        res["Bmso"] = np.vstack((res["Bmso"], Bmso[mask, :]))
                        res["Pmso"] = np.vstack((res["Pmso"], Pmso[mask, :]))
                except Exception as e:
                    print(f"Warning: failed to read {filename}: {e}")
        if res["time"].size == 0:
            return {"Bmso": None, "Pmso": None}
        sort_idx = np.argsort(res["time"])
        res["time"] = res["time"][sort_idx]
        res["Bmso"] = res["Bmso"][sort_idx, :]
        res["Pmso"] = res["Pmso"][sort_idx, :]
        B = ts_vec_xyz(
            res["time"],
            res["Bmso"],
            attrs={
                "name": "Magnetic field",
                "instrument": "MOMAG",
                "UNITS": "nT",
                "coordinates": "MSO",
                "resolution": config["resolution"],
            }
        )

        # Attempt to correct spacecraft position units automatically.
        if res["Pmso"].shape[0] > 0:
            if np.nanmax(norm(res["Pmso"])) > 200000:
                res["Pmso"] = res["Pmso"] / 3390.0

            if np.nanmin(norm(res["Pmso"])) < 20:
                res["Pmso"] = res["Pmso"] * 3390.0

        P = ts_vec_xyz(
            res["time"],
            res["Pmso"],
            attrs={
                "name": "Spacecraft position",
                "UNITS": "km",
                "coordinates": "MSO",
                "resolution": config["resolution"],
            }
        )

        return {"Bmso": B, "Pmso": P}
# %%
    # =========================================================================
    # CASE 2: MINPA MODE 1
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
        filelist = sorted(glob.glob(file_pattern))

        for filename in filelist:
            file_start, file_end = read_time_from_file(filename)

            if file_start <= end_time and file_end >= start_time:
                mat_data = loadmat(filename)

                time_array = irf_time(mat_data["time"], "datenum>datetime64").squeeze()
                mask = (time_array >= start_time) & (time_array <= end_time)

                if np.any(mask):
                    res["time"] = np.concatenate((res["time"], time_array[mask]))
                    res["energy"] = np.vstack((res["energy"], mat_data["energy"][mask, :]))
                    res["DEF"] = np.vstack((res["DEF"], mat_data["DEF"][mask, :, :, :, :]))
                    res["phi"] = np.asarray(mat_data["phi"]).squeeze()
                    res["theta"] = np.asarray(mat_data["theta"]).squeeze()
                    res["mass"] = np.asarray(mat_data["mass"]).squeeze()

        return res

# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    tint = ["2025-06-15T11:57:00", "2025-06-15T12:10:00"]

    # Example: Retrieve magnetic field
    B = get_data(tint, "B")
    print(B["Bmso"].data.shape)

    # Example: Retrieve MINPA Mode 1 DEF
    #minpa = get_data(tint, "minpa_mod1")
    #print(minpa["DEF"].shape)

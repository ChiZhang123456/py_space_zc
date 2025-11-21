"""
Venus Express Data Retrieval Module

Author: Chi Zhang (zc199508@bu.edu)
Last updated: 2025-11-11

This module provides functionality to extract and structure Venus Express spacecraft data
within a specified time interval and for a specific instrument or variable.

Supported variables include:
- "els_pad"   : Electron pitch angle distribution (PAD)
- "els_omni"  : Omni-directional electron energy spectra
- "B" / "B_1s": Magnetic field data (VEX MAG) in VSO coordinates

Dependencies:
- irfpy.vels, irfpy.vmag for data access
- py_space_zc (custom module) for unified data structures and plotting
- xarray for data packaging
- numpy, datetime, glob, re

Typical usage:

```python
from get_data import get_data

tint = ["2006-05-15T01:00:00", "2006-05-15T01:05:00"]
eomni = get_data(tint, "els_omni")
B     = get_data(tint, "B")
"""

import os
import numpy as np
import datetime
import glob
import re
import xarray as xr

import irfpy.vels.scidata as velsdata
import irfpy.vmag.scidata as vmagdata
import irfpy.vmag.scidata1s as vmagdata1s
import irfpy.vels.energy as vels_energy
from py_space_zc import plot, ts_vec_xyz, ts_spectr
from py_space_zc.vex import get_base_path
from py_space_zc.vex import read_els_pad

def get_data(tint, var):
    """
    Retrieve Venus Express data for the specified time interval and variable.

    Parameters
    ----------
    tint : list of str
        Time interval in ISO 8601 format. Example:
        ["2018-11-01T01:00:00", "2018-11-01T02:00:00"]

    var : str
        Variable to retrieve. Supported values:
        - "els_pad"  : Electron pitch angle data
        - "els_omni" : Omni-directional ELS spectrum
        - "B"        : Magnetic field data (MAG, 0.25 Hz)
        - "B_1s"     : Magnetic field data (same as "B" currently)

    Returns
    -------
        res : xarray.Dataset or ts_spectr or ts_vec_xyz
            A structured object containing the requested data.
    """

    # Convert string time to datetime
    t0 = datetime.datetime.fromisoformat(tint[0])
    t1 = datetime.datetime.fromisoformat(tint[1])

    # ======================
    # Case 1: els_pad
    # ======================
    if var == "els_pad":
        res = {
            "time": np.array([], dtype="datetime64[ns]"),
            "energy": np.empty((0, 127), dtype=float),
            "PSD": np.empty((0, 127, 18), dtype=float),
            "pa": None,
        }
        # Normalize and parse time interval
        start_time, end_time = np.datetime64(tint[0]), np.datetime64(tint[1])
        base_path = get_base_path()
        folder = os.path.join(base_path, "els_pad")
        pattern = "VExELSPADRG_???????.csv"
        search_pattern = os.path.join(folder, pattern.replace("?", "*"))
        all_files = sorted(glob.glob(search_pattern))

        # Date extraction helper
        def extract_doy(fname):
            match = re.search(r'(\d{7})', os.path.basename(fname))  # YYYYDOY
            if match:
                return np.datetime64(datetime.datetime.strptime(match.group(1), "%Y%j"))
            return None

        # Filter files within time range
        # Helper: check if a file with base time d overlaps [start_time, end_time)
        def file_overlaps(d, t_start, t_end):
            return not (d + np.timedelta64(1, 'D') <= t_start or d >= t_end)

        # Filter files within time range
        valid_files = []
        for f in all_files:
            d = extract_doy(f)
            if d is not None and file_overlaps(d, start_time, end_time):
                valid_files.append(f)
        for file in valid_files:
            try:
                time, energy_matrix, pa, pad_3d = read_els_pad(file)
                mask = (time >= start_time) & (time <= end_time)
                if np.any(mask):
                    res["time"] = np.concatenate((res["time"], time[mask]))
                    res["energy"] = np.vstack((res["energy"], energy_matrix[mask]))
                    res["PSD"] = np.vstack((res["PSD"], pad_3d[mask]))
                    if res["pa"] is None:
                        res["pa"] = pa
            except Exception as e:
                print(f"[Warning] Failed to read {file}: {e}")
                continue

        # Organize output
        n_time = len(res['time'])
        n_energy = res['energy'].shape[1]
        pad = xr.Dataset(
            {
                "data": (["time", "idx0", "idx1"], res["PSD"],),
                "energy": (["time", "idx0"], res['energy']),
                "pitchangle": (["idx1"], res["pa"]),
                "time": res['time'],
                "idx0": np.arange(n_energy),
                "idx1": np.arange(len(res["pa"])),
            }, )

        delta_angles = 10
        pad.attrs = {'species': 'e',
                     "delta_pitchangle_minus": delta_angles * 0.5,
                     "delta_pitchangle_plus": delta_angles * 0.5,
                     "UNITS": 's^3/m^6'}

        pad.data.attrs["UNITS"] = 's^3/m^6'
        return pad

#%%
    # ======================
    # Case 2: Magnetic field (VSO)
    # ======================
    elif var == "B":
        tb, bfield = vmagdata.get_magarray(t0, t1)
        tb_dt64 = np.array([np.datetime64(t, 'ns') for t in tb])
        B = ts_vec_xyz(tb_dt64, bfield, attrs={"name": "Magnetic field",
                                               "instrument": "Venus Express MAG",
                                               "UNITS":"nT",
                                               "coordinates":"VSO",
                                               "resolution":"0.25 Hz"})
        return B

    elif var == "B_1s":
        dc = vmagdata1s.DataCenterMag1s()
        time_list, mag_list = dc.get_array(t0, t1)
        mag_ma_array = np.ma.masked_array(mag_list)
        mag_array = mag_ma_array.filled(np.nan)
        tb_dt64 = np.array([np.datetime64(t, 'ns') for t in time_list])
        B = ts_vec_xyz(tb_dt64, mag_array, attrs={"name": "Magnetic field",
                                               "instrument": "Venus Express MAG",
                                               "UNITS":"nT",
                                               "coordinates":"VSO",
                                               "resolution":"0.25 Hz"})
        return B

    # %%
    # ======================
    # Case 3: Omni-directional electron spectrum
    # ======================
    elif var == "els_omni":
        el_enetbl = vels_energy.get_default_table_128()
        t_els, elsmatrix = velsdata.get_counts(t0, t1)
        els_omni_count = np.nansum(elsmatrix, axis = 1).T
        tels_dt64 = np.array([np.datetime64(t, 'ns') for t in t_els])
        el_enetbl = np.flip(el_enetbl)
        els_omni_count = np.flip(els_omni_count, axis = 1)
        res = ts_spectr(tels_dt64, el_enetbl[1:], els_omni_count[:,1:],
                        attrs={"name": "VEX ELS omni count",
                               "UNITS":"None",
                               "resolution":"0.25 Hz"})
        return res



# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    tint = ["2006-05-15T01:00:00", "2006-05-15T01:05:00"]

    # Example: Retrieve electron pitch angle data
    eomni = get_data(tint, "els_omni")
    B = get_data(tint, "B")

    from py_space_zc import plot
    import matplotlib.pyplot as plt
    ax,_ = plot.plot_spectr(None, eomni, cscale='log', yscale='log',cmap = 'Spectral_r',)
    ax.set_ylim(1.0, 30000.0)
    plt.show()

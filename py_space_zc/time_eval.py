import numpy as np
import xarray as xr
from pyrfu.pyrf import iso86012datetime64
from datetime import datetime

def time_eval(inp: xr.DataArray | xr.Dataset, time: str | np.datetime64 | datetime):
    """
    Evaluate the data at the closest time to the input time.

    Parameters
    ----------
    inp : xarray.DataArray or xarray.Dataset
        Input time series.
    time : str or np.datetime64 or datetime.datetime
        Target time to evaluate. Can be ISO 8601 string, np.datetime64, or datetime object.

    Returns
    -------
    out : xarray.DataArray or xarray.Dataset
        Data at the closest time to the input time.
    """
    # --- Convert time to np.datetime64 ---
    if isinstance(time, str):
        time = np.datetime64(time)
    elif isinstance(time, datetime):
        time = np.datetime64(time)
    elif not isinstance(time, np.datetime64):
        raise TypeError("time must be str, datetime, or np.datetime64!")

    # --- Find nearest time index ---
    times = inp.time.data
    idx = np.argmin(np.abs(times - time))
    t_nearest = times[idx]

    # --------- Dataset case ----------
    if isinstance(inp, xr.Dataset):
        out_dict = {}

        # Variables
        for var in inp.data_vars:
            if "time" in inp[var].dims:
                out_dict[var] = inp[var].sel(time=t_nearest)
            else:
                out_dict[var] = inp[var]

        # Coordinates
        coords_dict = {}
        for cname, cval in inp.coords.items():
            if "time" in cval.dims:
                coords_dict[cname] = cval.sel(time=t_nearest)
            else:
                coords_dict[cname] = cval

        # Attributes
        out_attrs = {}
        for k, v in inp.attrs.items():
            if isinstance(v, np.ndarray) and v.shape[0] == len(times):
                out_attrs[k] = v[idx]
            else:
                out_attrs[k] = v

        return xr.Dataset(data_vars=out_dict, coords=coords_dict, attrs=out_attrs)

    # --------- DataArray case ----------
    elif isinstance(inp, xr.DataArray):
        coords_dict = {}
        for cname, cval in inp.coords.items():
            if "time" in cval.dims:
                coords_dict[cname] = cval.sel(time=t_nearest)
            else:
                coords_dict[cname] = cval

        return xr.DataArray(
            data=inp.sel(time=t_nearest).data,
            coords=coords_dict,
            dims=[d for d in inp.dims if d != "time"],
            attrs=inp.attrs
        )

    else:
        raise TypeError("Input must be xarray.Dataset or xarray.DataArray")

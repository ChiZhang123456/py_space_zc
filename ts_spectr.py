"""
Adapted from pyrfu's ts_skymap module:
(https://github.com/louis-richard/irfu-python), licensed under the MIT License

Original code licensed under the MIT License.
Modified by Chi Zhang for compatibility with py_space_zc.
"""

# 3rd party imports
import numpy as np
import xarray as xr
def ts_spectr(time, ener, data, comp_name: str = "energy", attrs: dict = None):
    r"""Create a time series containing a spectrum

    Parameters
    ----------
    time : numpy.ndarray
        Array of times.
    ener : numpy.ndarray
        Y value of the spectrum (energies, frequencies, etc.)
    data : numpy.ndarray
        Data of the spectrum.
    attrs : dict, Optional
        Attributes of the data list.

    Returns
    -------
    out : xarray.DataArray
        Time series of a spectrum

    """

    # Check input type
    assert isinstance(time, np.ndarray), "time must be a numpy.ndarray"
    assert isinstance(ener, np.ndarray), "ener must be a numpy.ndarray"
    assert isinstance(data, np.ndarray), "data must be a numpy.ndarray"

    # Check input shape must be (n, m, )
    assert data.ndim == 2, "Input must be a spectrum"
    assert len(time) == data.shape[0], "len(time) and data.shape[0] must be equal"

    if ener.ndim == 1:
        assert len(ener) == data.shape[1], "len(ener) and data.shape[1] must be equal"
    elif ener.ndim == 2:
        assert ener.shape == data.shape, (f"2D ener shape {ener.shape} must match data shape {data.shape}")
    if attrs is None or not isinstance(attrs, dict):
        attrs = {}

    if ener.ndim == 1:
        out = xr.DataArray(data,
                           coords={"time": time, comp_name: ener},
                           dims=["time", comp_name],
                           attrs=attrs)
    else:
        # For 2D energy, define it as an auxiliary coordinate
        out = xr.DataArray(data,
                           coords={"time": time,
                                   comp_name: (["time", comp_name], ener)},
                           dims=["time", comp_name],
                           attrs=attrs)
    return out

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3rd party imports
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

def plot_scatter(axis, inp, **kwargs):
    r"""Scatter plot of time series.

    Parameters
    ----------
    axis : matplotlib.axes._axes.Axes or None
        Target axis where to plot `inp`. If None, creates a new figure with a single axis.
    inp : xarray.DataArray
        Time series to plot. The x-axis is taken from `inp.time`.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments forwarded to `matplotlib.axes.Axes.scatter`, e.g.
        `s`, `marker`, `c`, `alpha`, etc.

    Returns
    -------
    axis : matplotlib.axes._axes.Axes
        Axis with the scatter artists added.

    Notes
    -----
    - If `inp.data` is 1-D, a single scatter is drawn.
    - If `inp.data` is 2-D (time, n_series), a scatter is drawn per column.
    - If `inp.data` is 3-D (time, m, n), it is reshaped to (time, m*n) and a scatter is drawn per column.
    - Time axes are auto-formatted using `ConciseDateFormatter` when dtype is datetime64[ns].
    """

    # Create or validate axis
    if axis is None:
        _, axis = plt.subplots(1)
    else:
        if not isinstance(axis, mpl.axes.Axes):
            raise TypeError("axis must be a matplotlib.axes._axes.Axes")

    # Validate input type
    if not isinstance(inp, xr.DataArray):
        raise TypeError("inp must be an xarray.DataArray object!")

    # Normalize data shape to (time, n_series)
    if inp.data.ndim < 3:
        data = inp.data
    elif inp.data.ndim == 3:
        data = np.reshape(inp.data, (inp.shape[0], inp.shape[1] * inp.shape[2]))
    else:
        raise NotImplementedError(
            f"plot_scatter cannot handle {inp.data.ndim} dimensional data"
        )

    time = inp.time

    # Draw scatters
    # If data is 1-D: single scatter
    # If data is 2-D: one scatter per column
    if data.ndim == 1:
        axis.scatter(time, data, **kwargs)
    elif data.ndim == 2:
        for i in range(data.shape[1]):
            axis.scatter(time, data[:, i], **kwargs)
    else:
        # Should not reach here given the checks above
        raise RuntimeError("Unexpected data dimensionality after normalization.")

    # Time axis formatting (if datetime64[ns])
    if time.dtype == "<M8[ns]":
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    # Styling: grid and y tick locator similar to plot_line
    axis.grid(True, which="major", linestyle="-", linewidth="0.2", c="0.5")
    axis.yaxis.set_major_locator(mticker.MaxNLocator(4))
    axis.set_xlim(np.datetime64(inp.time.data[0]),np.datetime64(inp.time.data[-1]))
    axis.tick_params(axis='both', direction='in')

    return axis

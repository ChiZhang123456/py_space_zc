#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_spectr(
    axis,
    inp,
    yscale: str = "linear",
    cscale: str = "linear",
    clim: list = None,
    cmap: str = None,
    colorbar: str = "right",
    **kwargs,
):
    """
    Plot a 2D spectrogram using pcolormesh from an xarray.DataArray.

    Parameters
    ----------
    axis : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis will be created.
    inp : xarray.DataArray
        Input 2D DataArray (typically time vs energy or frequency).
    yscale : str, optional
        Y-axis scaling, "linear" or "log". Default is "linear".
    cscale : str, optional
        Color scale scaling, "linear" or "log". Default is "linear".
    clim : list of float, optional
        Color limits [vmin, vmax]. If None, determined automatically.
    cmap : str, optional
        Colormap name. Default is matplotlib default.
    colorbar : str, optional
        Placement of colorbar: "right", "top", or "none".
    **kwargs :
        Additional keyword arguments. "pad" is used for colorbar padding.

    Returns
    -------
    out : tuple or axis
        Returns (axis, cax) if colorbar is added; otherwise returns axis only.
    """

    # Create axis if none is provided
    if axis is None:
        fig, axis = plt.subplots(1)
    else:
        fig = plt.gcf()

    # Get and configure colormap
    if not cmap or isinstance(cmap, str):
        cmap = mpl.colormaps.get_cmap(cmap).copy()
     #   cmap.set_bad(color="grey")  # Show NaNs in gray
    else:
        raise TypeError(
            "cmap must be a string. "
            "To add a custom colormap use mpl.colormaps.register(custom)."
        )

    # Build color scale options
    if cscale == "log":
        if clim is not None and isinstance(clim, list):
            options = {
                "norm": mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1]),
                "cmap": cmap,
            }
        else:
            options = {"norm": mpl.colors.LogNorm(), "cmap": cmap}
    else:
        if clim is not None and isinstance(clim, list):
            options = {"cmap": cmap, "vmin": clim[0], "vmax": clim[1]}
        else:
            options = {"cmap": cmap}

    # Extract axes coordinates
    x_data = inp.coords[inp.dims[0]]
    y_data = inp.coords[inp.dims[1]]

    # Choose plotting approach based on y_data dimensionality
    if y_data.ndim == 1:
        z_data = inp.data.T.copy()
        if cscale == "log":
            z_data[z_data <= 0] = np.nan
        image = axis.pcolormesh(
            x_data.data,
            y_data.data,
            z_data,
            rasterized=True,
            shading="auto",
            **options,
        )

    elif y_data.ndim == 2:
        assert y_data.shape == inp.data.shape, "y_data and data must have same shape"

        t_grid = np.tile(x_data.data[:, None], (1, y_data.shape[1]))
        z_data = inp.data.copy()
        if cscale == "log":
            z_data[z_data <= 0] = np.nan

        image = axis.pcolormesh(
            t_grid,
            y_data.data,
            z_data,
            rasterized=True,
            shading="auto",
            **options,
        )

    else:
        raise ValueError("y_data must be 1D or 2D")

    # Format time axis if datetime
    if x_data.dtype == "<M8[ns]":
        locator = mpl.dates.AutoDateLocator(minticks=5, maxticks=9)
        formatter = mpl.dates.ConciseDateFormatter(locator)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    # Set y-axis scale
    if yscale == "log":
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))

    # Set y-axis limits
    axis.set_axisbelow(False)
    min_ydata = np.floor(np.nanmin(y_data.data)).astype(np.float32)+0.1
    max_ydata = np.ceil(np.nanmax(y_data.data)).astype(np.float32)
    axis.set_ylim(min_ydata, max_ydata)

    # Plot colorbar
    if colorbar.lower() == "right":
        pad = kwargs.get("pad", 0.005)
        pos = axis.get_position()
        cax = fig.add_axes([pos.x0 + pos.width + pad, pos.y0, 0.005, pos.height])
        plt.colorbar(mappable=image, cax=cax, ax=axis, orientation="vertical")

        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")
        cax.set_axisbelow(False)

        if cscale == "log":
            cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))
        else:
            cax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

        out = (axis, cax)

    elif colorbar.lower() == "top":
        pad = kwargs.get("pad", 0.01)
        pos = axis.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 + pos.height + pad, pos.width, 0.01])
        plt.colorbar(mappable=image, cax=cax, ax=axis, orientation="horizontal")

        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.set_axisbelow(False)

        if cscale == "log":
            cax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=4))
        else:
            cax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

        out = (axis, cax)

    elif colorbar.lower() == "none":
        out = axis

    else:
        raise NotImplementedError("colorbar must be: 'right', 'top', or 'none'")

    return out

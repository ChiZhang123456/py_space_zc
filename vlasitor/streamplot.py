from __future__ import annotations

# === Standard library imports ===
import numpy as np
from numpy.typing import ArrayLike
import math
import warnings
from typing import Callable
from collections import namedtuple
from enum import Enum

# === PyVlasiator imports ===
from pyvlasiator.vlsv import Vlsv
from pyvlasiator.vlsv.variables import RE, RMERCURY
from pyvlasiator import plot as vplt


# === Named tuple for plot metadata ===
# Stores metadata for slices and labels (used in color plots, etc.)
PlotArgs = namedtuple(
    "PlotArgs",
    [
        "axisunit",   # Unit of the plot axes (e.g., EARTH, MERCURY)
        "sizes",      # 2D grid size in pixel counts
        "plotrange",  # Physical extent (xmin, xmax, ymin, ymax)
        "origin",     # Slice position along the normal axis
        "idlist",     # AMR cell IDs used in slice
        "indexlist",  # Raw cell indices from file
        "str_title",  # Plot title
        "strx",       # X-axis label
        "stry",       # Y-axis label
        "cb_title",   # Colorbar label
    ],
)


def _getdim2d(ncells: tuple, maxamr: int, normal: int):
    """
    Compute the 2D slice dimensions based on AMR level and normal direction.

    Parameters
    ----------
    ncells : tuple of int
        Number of cells in x/y/z.
    maxamr : int
        Maximum refinement level.
    normal : int
        The axis normal to the slice plane: 0=x, 1=y, 2=z.

    Returns
    -------
    dims : tuple of int
        (nx, ny) dimensions of the 2D grid slice.
    """
    ratio = 2**maxamr  # AMR resolution multiplier

    # Select in-plane axes
    if normal == 0:
        i1, i2 = 1, 2
    elif normal == 1:
        i1, i2 = 0, 2
    elif normal == 2:
        i1, i2 = 0, 1
    else:
        raise ValueError("normal must be 0, 1, or 2")

    dims = (ncells[i1] * ratio, ncells[i2] * ratio)
    return dims


def streamplot(
    meta: Vlsv,
    var: str,
    ax=None,
    comp: str = "xy",
    axisunit: vplt.AxisUnit = vplt.AxisUnit.EARTH,
    origin: float = 0.0,
    **kwargs,
) -> matplotlib.streamplot.StreamplotSet:
    """
    Generate a 2D streamplot of a vector field from a VLSV dataset.

    Parameters
    ----------
    meta : Vlsv
        Vlasiator metadata object.
    var : str
        Name of the vector variable (e.g., 'vg_v', 'fg_b').
    ax : matplotlib.axes.Axes, optional
        Target axis to plot into. If None, a new figure is created.
    comp : str, optional
        Slice plane component, e.g., 'xy', 'xz', 'yz'. Default is 'xy'.
    axisunit : AxisUnit, optional
        Unit for physical axes. Default is Earth radii.
    origin : float, optional
        Slice location along the direction normal to the selected plane.
    **kwargs :
        Additional arguments passed to `ax.streamplot()`.

    Returns
    -------
    stream : matplotlib.streamplot.StreamplotSet
        Resulting streamplot object.
    """
    # Extract vector and coordinates
    X, Y, v1, v2 = set_vector(meta, var, comp, axisunit, origin)
    # Set figure or reuse existing one
    fig, ax = set_figure(ax, **kwargs)
    # Call streamplot
    s = ax.streamplot(X, Y, v1, v2, **kwargs)
    return s


def set_vector(
    meta: Vlsv, var: str, comp: str, axisunit: vplt.AxisUnit, origin: float = 0.0
):
    """
    Extract and slice a 3D vector field into a 2D plane.

    Parameters
    ----------
    meta : Vlsv
        Vlsv object with metadata and grid structure.
    var : str
        Variable name to load (must be vector type with 3 components).
    comp : str
        Plane to extract (e.g., 'xy', 'xz', 'yz').
    axisunit : AxisUnit
        Unit system to use (e.g., EARTH, MERCURY).
    origin : float
        Slice location along the normal axis (e.g., z=0 for 'xy').

    Returns
    -------
    x, y : ndarray
        Coordinate grid in physical units.
    v1, v2 : ndarray
        2D field components in the slice plane.

    Raises
    ------
    ValueError
        If variable is not 3-component or not in a known format.
    """
    ncells = meta.ncells
    maxamr = meta.maxamr
    coordmin, coordmax = meta.coordmin, meta.coordmax

    # Determine slice plane and indices
    if "x" in comp:
        v1_ = 0
        if "y" in comp:
            dir = 2
            v2_ = 1
            sizes = _getdim2d(ncells, maxamr, 2)
            plotrange = (coordmin[0], coordmax[0], coordmin[1], coordmax[1])
        else:
            dir = 1
            v2_ = 2
            sizes = _getdim2d(ncells, maxamr, 1)
            plotrange = (coordmin[0], coordmax[0], coordmin[2], coordmax[2])
    else:
        dir = 0
        v1_, v2_ = 1, 2
        sizes = _getdim2d(ncells, maxamr, 0)
        plotrange = (coordmin[1], coordmax[1], coordmin[2], coordmax[2])

    # Read variable
    data = meta.read_variable(var)

    if not var.startswith("fg_"):  # Vlasov AMR grid
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("Expected a 3-component vector variable.")
        if maxamr == 0:
            # Uniform grid, reshape directly

            data = data.reshape((sizes[1], sizes[0], 3))
            v1 = data[:, :, v1_]
            v2 = data[:, :, v2_]
            # data = np.squeeze(data)
            # v1 = np.transpose(data[:, :, v1_])
            # v2 = np.transpose(data[:, :, v2_])

        else:
            # AMR slice extraction
            sliceoffset = origin - coordmin[dir]
            idlist, indexlist = meta.getslicecell(sliceoffset, dir, coordmin[dir], coordmax[dir])
            v2D = data[indexlist, :]
            v1 = meta.refineslice(idlist, v2D[:, v1_], dir)
            v2 = meta.refineslice(idlist, v2D[:, v2_], dir)
    else:  # Field grid (regular 3D array)
        data = np.squeeze(data)
        v1 = np.transpose(data[:, :, v1_])
        v2 = np.transpose(data[:, :, v2_])

    # Compute axes in physical units
    x, y = get_axis(axisunit, plotrange, sizes)
    return x, y, v1, v2


def set_figure(ax, figsize: tuple = (10, 6), **kwargs) -> tuple:
    """
    Create or return a Matplotlib figure and axis for plotting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If provided, use this axis. If None, create a new one.
    figsize : tuple of float
        Figure size in inches. Default is (10, 6).
    **kwargs :
        Extra options forwarded to `plt.figure()` if needed.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    # If a figure is passed, use it; else create or reuse current figure
    fig = kwargs.pop("figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize))
    if ax is None:
        ax = fig.gca()
    return fig, ax

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import math
import warnings
from typing import Callable
from collections import namedtuple
from enum import Enum
from pyvlasiator.vlsv import Vlsv
from pyvlasiator.vlsv.reader import _getdim2d
from pyvlasiator.vlsv.variables import RE, RMERCURY
from pyvlasiator import plot as vplt

class ColorScale(Enum):
    """
    Represents the available color scales for data visualization.

    Attributes:
        - Linear (1): A linear color scale, where colors are evenly distributed across the data range.
        - Log (2): A logarithmic color scale, suitable for data with a wide range of values, where smaller values are more emphasized.
        - SymLog (3): A symmetric logarithmic color scale, similar to Log but with symmetry around zero, useful for data with both positive and negative values.
    """

    Linear = 1
    Log = 2
    SymLog = 3


# Plotting arguments
PlotArgs = namedtuple(
    "PlotArgs",
    [
        "axisunit",
        "sizes",
        "plotrange",
        "origin",
        "idlist",
        "indexlist",
        "str_title",
        "strx",
        "stry",
        "cb_title",
    ],
)


def prep2d(meta: Vlsv, var: str, comp: int = -1):
    """
    Obtain data of `var` for 2D plotting. Use `comp` to select vector components.

    Parameters
        ----------
        meta : Vlsv
            Metadata corresponding to the file.
        var : str
            Name of the variable.
        comp : int
            Vector component. -1 refers to the magnitude of the vector.
        Returns
        -------
        numpy.ndarray
    """

    dataRaw = _getdata2d(meta, var)

    if dataRaw.ndim == 3:
        if comp != -1:
            data = dataRaw[:, :, comp]
        else:
            data = np.linalg.norm(dataRaw, axis=2)
        if var.startswith("fg_"):
            data = np.transpose(data)
    else:
        data = dataRaw

    return data

def prep2dslice(meta: Vlsv, var: str, dir: int, comp: int, pArgs: PlotArgs):
    origin = pArgs.origin
    idlist = pArgs.idlist
    indexlist = pArgs.indexlist

    data3D = meta.read_variable(var)

    if var.startswith("fg_") or data3D.ndim > 2:  # field or derived quantities, fsgrid
        ncells = meta.ncells * 2**meta.maxamr
        if not dir in (0, 1, 2):
            raise ValueError(f"Unknown normal direction {dir}")

        sliceratio = (origin - meta.coordmin[dir]) / (
            meta.coordmax[dir] - meta.coordmin[dir]
        )
        if not (0.0 <= sliceratio <= 1.0):
            raise ValueError("slice plane index out of bound!")
        # Find the cut plane index for each refinement level
        icut = int(np.floor(sliceratio * ncells[dir]))
        if dir == 0:
            if comp != -1:
                data = data3D[icut, :, :, comp]
            else:
                data = np.linalg.norm(data3D[icut, :, :, :], axis=2)
        elif dir == 1:
            if comp != -1:
                data = data3D[:, icut, :, comp]
            else:
                data = np.linalg.norm(data3D[:, icut, :, :], axis=2)
        elif dir == 2:
            if comp != -1:
                data = data3D[:, :, icut, comp]
            else:
                data = np.linalg.norm(data3D[:, :, icut, :], axis=2)
    else:  # moments, dccrg grid
        # vlasov grid, AMR
        if data3D.ndim == 1:
            data2D = data3D[indexlist]

            data = meta.refineslice(idlist, data2D, dir)
        elif data3D.ndim == 2:
            data2D = data3D[indexlist, :]

            if comp in (0, 1, 2):
                slice = data2D[:, comp]
                data = meta.refineslice(idlist, slice, dir)
            elif comp == -1:
                datax = meta.refineslice(idlist, data2D[:, 0], dir)
                datay = meta.refineslice(idlist, data2D[:, 1], dir)
                dataz = meta.refineslice(idlist, data2D[:, 2], dir)
                data = np.fromiter(
                    (np.linalg.norm([x, y, z]) for x, y, z in zip(datax, datay, dataz)),
                    dtype=float,
                )
            else:
                slice = data2D[:, comp]
                data = meta.refineslice(idlist, slice, dir)

    return data


def configure_plot(
    c: matplotlib.cm.ScalarMappable,  # Assuming c is a colormap or similar
    ax: matplotlib.pyplot.Axes,
    plot_args: PlotArgs,
    ticks: list,
    add_colorbar: bool = True,
):
    """
    Configures plot elements based on provided arguments.
    """

    import matplotlib.pyplot as plt

    title = plot_args.str_title
    x_label = plot_args.strx
    y_label = plot_args.stry
    colorbar_title = plot_args.cb_title

    # Add colorbar if requested
    if add_colorbar:
        cb = plt.colorbar(c, ax=ax, ticks=ticks, fraction=0.04, pad=0.02)
        if colorbar_title:
            cb.ax.set_ylabel(colorbar_title)
        cb.ax.tick_params(direction="in")

    # Set title and labels
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")

    # Style borders and ticks
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.tick_params(axis="both", which="major", width=2.0, length=3)


def set_args(
    meta: Vlsv,
    var: str,
    axisunit: AxisUnit = vplt.AxisUnit.EARTH,
    dir: int = -1,
    origin: float = 0.0,
) -> PlotArgs:
    """
    Set plot-related arguments of `var` in `axisunit`.

    Parameters
    ----------
    var : str
        Variable name from the VLSV file.
    axisunit : AxisUnit
        Unit of the axis.
    dir : int
        Normal direction of the 2D slice, 0 for x, 1 for y, and 2 for z.
    origin : float
        Origin of the 2D slice.

    Returns
    -------
    PlotArgs

    See Also
    --------
    :func:`pcolormesh`
    """

    ncells, coordmin, coordmax = meta.ncells, meta.coordmin, meta.coordmax

    if dir == 0:
        seq = (1, 2)
    elif dir == 1 or (ncells[1] == 1 and ncells[2] != 1):  # polar
        seq = (0, 2)
        dir = 1
    elif dir == 2 or (ncells[2] == 1 and ncells[1] != 1):  # ecliptic
        seq = (0, 1)
        dir = 2
    else:
        raise ValueError("1D data detected. Please use 1D plot functions.")

    plotrange = (coordmin[seq[0]], coordmax[seq[0]], coordmin[seq[1]], coordmax[seq[1]])
    axislabels = tuple(("X", "Y", "Z")[i] for i in seq)
    # Scale the sizes to the highest refinement level for data to be refined later
    sizes = tuple(ncells[i] << meta.maxamr for i in seq)

    if dir == -1:
        idlist, indexlist = np.empty(0, dtype=int), np.empty(0, dtype=int)
    else:
        sliceoffset = origin - coordmin[dir]
        idlist, indexlist = meta.getslicecell(
            sliceoffset, dir, coordmin[dir], coordmax[dir]
        )

    if axisunit == vplt.AxisUnit.EARTH:
        unitstr = r"$R_E$"
    elif axisunit == vplt.AxisUnit.SI:
        unitstr = "m"
    elif axisunit == vplt.AxisUnit.MERCURY:
        unitstr = r"$R_M$"
    elif axisunit == vplt.AxisUnit.KSI:
        unitstr = "km"
    strx = axislabels[0] + " [" + unitstr + "]"
    stry = axislabels[1] + " [" + unitstr + "]"

    str_title = f"t={meta.time:4.1f}s"

    datainfo = meta.read_variable_meta(var)

    if not datainfo.variableLaTeX:
        cb_title = datainfo.variableLaTeX + " [" + datainfo.unitLaTeX + "]"
    else:
        cb_title = ""

    return PlotArgs(
        axisunit,
        sizes,
        plotrange,
        origin,
        idlist,
        indexlist,
        str_title,
        strx,
        stry,
        cb_title,
    )

def set_colorbar(
    colorscale: ColorScale = ColorScale.Linear,
    v1: float = np.nan,
    v2: float = np.nan,
    data: np.ndarray = np.array([1.0]),
    linthresh: float = 1.0,
    linscale: float = 0.03,
):
    """
    Creates a color normalization object and tick values for a colorbar.

    Parameters
    ----------
    colorscale: ColorScale, optional
        The type of color scale to use. Can be 'Linear', 'Log', or 'SymLog'.
        Defaults to 'Linear'.
    v1: float, optional
        The minimum value for the colorbar. Defaults to np.nan, which means
        it will be inferred from the data.
    v2: float, optional
        The maximum value for the colorbar. Defaults to np.nan, which means
        it will be inferred from the data.
    data: np.ndarray, optional
        The data to use for inferring the colorbar limits if v1 and v2 are
        not provided. Defaults to np.array([1.0]).
    linthresh: float, optional
        The threshold value for symmetric log color scales. Defaults to 1.0.
    linscale: float, optional
        A scaling factor for linear regions in symmetric log color scales.
        Defaults to 0.03.

    Returns
    -------
    tuple
        A tuple containing:
            - norm: A Matplotlib color normalization object for the colorbar.
            - ticks: A list of tick values for the colorbar.

    Raises
    ------
    ValueError
        If an invalid colorscale type is provided.
    """
    import matplotlib

    vmin, vmax = set_plot_limits(v1, v2, data, colorscale)
    if colorscale == ColorScale.Linear:
        levels = matplotlib.ticker.MaxNLocator(nbins=255).tick_values(vmin, vmax)
        norm = matplotlib.colors.BoundaryNorm(levels, ncolors=256, clip=True)
        ticks = matplotlib.ticker.LinearLocator(numticks=9)
    elif colorscale == ColorScale.Log:  # logarithmic
        norm = matplotlib.colors.LogNorm(vmin, vmax)
        ticks = matplotlib.ticker.LogLocator(base=10, subs=range(0, 9))
    else:  # symmetric log
        norm = matplotlib.colors.SymLogNorm(
            linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax, base=10
        )
        ticks = matplotlib.ticker.SymmetricalLogLocator(linthresh=linthresh, base=10)

    return norm, ticks

def set_plot_limits(
    vmin: float,
    vmax: float,
    data: np.ndarray,
    colorscale: ColorScale = ColorScale.Linear,
) -> tuple:
    """
    Calculates appropriate plot limits based on data and colorscale.
    """

    if colorscale in (ColorScale.Linear, ColorScale.SymLog):
        vmin = vmin if not math.isinf(vmin) else np.nanmin(data)
        vmax = vmax if not math.isinf(vmax) else np.nanmax(data)
    else:  # Logarithmic colorscale
        positive_data = data[data > 0.0]  # Exclude non-positive values
        vmin = vmin if not math.isinf(vmin) else np.min(positive_data)
        vmax = vmax if not math.isinf(vmax) else np.max(positive_data)

    return vmin, vmax

def get_axis(axisunit: AxisUnit, plotrange: tuple, sizes: tuple) -> tuple:
    """
    Generates the 2D domain axis coordinates, potentially applying Earth radius scaling.

    Parameters
    ----------
    axisunit: AxisUnit
        The unit system for the plot axes.
    plotrange: tuple
        A tuple containing the minimum and maximum values for both axes (xmin, xmax, ymin, ymax).
    sizes: tuple
        A tuple containing the number of points for each axis (nx, ny).

    Returns
    -------
    tuple
        A tuple containing the x and y axis coordinates.
    """

    if axisunit == vplt.AxisUnit.EARTH:
        scale_factor = 1.0 / RE
    elif axisunit == vplt.AxisUnit.MERCURY:
        scale_factor = 1.0 / RMERCURY
    elif axisunit == vplt.AxisUnit.KSI:
        scale_factor = 1.0 / 1e3
    else:  # SI
        scale_factor = 1.0

    start = tuple(s * scale_factor for s in plotrange[:2])
    stop = tuple(s * scale_factor for s in plotrange[2:])

    # Vectorized generation of coordinates for efficiency
    x, y = np.linspace(*start, num=sizes[0]), np.linspace(*stop, num=sizes[1])

    return x, y

def set_figure(ax, figsize: tuple = (10, 6), **kwargs) -> tuple:
    """
    Sets up a Matplotlib figure and axes for plotting.

    Parameters
    ----------
    ax: matplotlib.axes.Axes, optional
        An existing axes object to use for plotting. If not provided, a new
        figure and axes will be created.
    figsize: tuple, optional
        The desired figure size in inches, as a tuple (width, height).
        Defaults to (10, 6).
    **kwargs
        Additional keyword arguments passed to Matplotlib's figure() function
        if a new figure is created.

    Returns
    -------
    tuple
        A tuple containing:
            - fig: The Matplotlib figure object.
            - ax: The Matplotlib axes object.
    """

    import matplotlib.pyplot as plt

    fig = kwargs.pop(
        "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
    )
    if ax is None:
        ax = fig.gca()

    return fig, ax


def pcolormesh(
    self: Vlsv,
    var: str = "",
    axisunit: AxisUnit = vplt.AxisUnit.EARTH,
    colorscale: ColorScale = ColorScale.Linear,
    addcolorbar: bool = True,
    vmin: float = float("-inf"),
    vmax: float = float("inf"),
    extent: list = [0.0, 0.0, 0.0, 0.0],
    comp: int = -1,
    ax=None,
    figsize: tuple[float, float] | None = None,
    **kwargs,
):
    """
    Plots 2D VLSV data using pcolormesh.

    Parameters
    ----------
    var : str
        Name of the variable to plot from the VLSV file.
    axisunit : AxisUnit
        Unit of the axis, `AxisUnit.EARTH` or `AxisUnit.SI`.
    addcolorbar : bool
        Add colorbar to the right.
    colorscale: ColorScale
        Color scale of the data, `ColorScale.Linear`, `ColorScale.Log`, or `ColorScale.SymLog`.
    extent : list
        Extent of the domain (WIP).
    comp : int
        Vector composition of data, -1 is magnitude, 0 is x, 1 is y, and 2 is z.
    ax : matplotlib.axes._axes.Axes, optional
        Axes object to plot on. If not provided, a new figure and axes
        will be created using `set_figure`.
    figsize : tuple[float, float], optional
        Size of the figure in inches. Only used if a new figure is created.
    **kwargs
        Additional keyword arguments passed to `ax.pcolormesh`.

    Returns
    -------
    matplotlib.figure.Figure
        The created or existing figure object.

    Raises
    ------
    ValueError
        If the specified variable is not found in the VLSV file.

    Examples
    --------
    >>> vlsv_file = Vlsv("my_vlsv_file.vlsv")
    >>> # Plot density on a new figure
    >>> fig = vlsv_file.pcolormesh("proton/vg_rho")
    >>> # Plot velocity on an existing axes
    >>> ax = ...  # Existing axes object
    >>> fig = vlsv_file.pcolormesh("proton/vg_v", ax=ax)
    """

    fig, ax = set_figure(ax, figsize, **kwargs)

    return _plot2d(
        self,
        ax.pcolormesh,
        var=var,
        ax=ax,
        comp=comp,
        axisunit=axisunit,
        colorscale=colorscale,
        addcolorbar=addcolorbar,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        **kwargs,
    )


def _plot2d(
    meta: Vlsv,
    plot_func: Callable,
    var: str = "",
    axisunit: AxisUnit = vplt.AxisUnit.EARTH,
    colorscale: ColorScale = ColorScale.Linear,
    addcolorbar: bool = True,
    vmin: float = float("-inf"),
    vmax: float = float("inf"),
    extent: list = [0.0, 0.0, 0.0, 0.0],
    comp: int = -1,
    ax=None,
    **kwargs,
):
    """
    Plot 2d data.

    Parameters
    ----------
    var : str
        Variable name from the VLSV file.

    Returns
    -------

    """

    if not meta.has_variable(var):
        raise ValueError(f"Variable {var} not found in the file")

    if meta.ndims() == 3 or meta.maxamr > 0:
        # check if origin and normal exist in kwargs
        normal = kwargs["normal"] if "normal" in kwargs else 1
        origin = kwargs["origin"] if "origin" in kwargs else 0.0
        kwargs.pop("normal", None)
        kwargs.pop("origin", None)

        pArgs = set_args(meta, var, axisunit, normal, origin)
        data = prep2dslice(meta, var, normal, comp, pArgs)
    else:
        pArgs = set_args(meta, var, axisunit)
        data = prep2d(meta, var, comp)

    x1, x2 = get_axis(pArgs.axisunit, pArgs.plotrange, pArgs.sizes)

    if var in ("fg_b", "fg_e", "vg_b_vol", "vg_e_vol") or var.endswith("vg_v"):
        _fillinnerBC(data)

    norm, ticks = set_colorbar(colorscale, vmin, vmax, data)

    range1 = range(
        np.searchsorted(x1, extent[0]), np.searchsorted(x1, extent[1], side="right")
    )
    range2 = range(
        np.searchsorted(x2, extent[2]), np.searchsorted(x2, extent[3], side="right")
    )

    c = plot_func(x1, x2, data, **kwargs)

    configure_plot(c, ax, pArgs, ticks, addcolorbar)

    return c



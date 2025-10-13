import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Sequence
from py_space_zc import cone_clock_angle, ts_scalar, plot

def plot_clock_angle(ax: Optional[plt.Axes],
                     B,
                     label: Optional[str] = None,
                     color: str = "black",
                     linewidth: float = 1.0,
                     linestyle: str = "-",
                     style: str = "line",          # "line" or "dot"
                     markersize: float = 4.0,      # used if style == "dot"
                     ylim: Tuple[float, float] = (0.0, 360.0),
                     yticks: Optional[Sequence[float]] = (90, 180, 270),
                     set_ylabel: bool = True) -> plt.Axes:
    """
    Plot the magnetic-field clock angle from an xarray.DataArray time series.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis. If None, a new figure/axis will be created.
    B : xarray.DataArray
        Magnetic field time series, shape (N, 3), with .time and .data.
    label : str, optional
        Legend label.
    color : str, optional
        Line/marker color.
    linewidth : float, optional
        Line width (used when style == "line").
    linestyle : str, optional
        Line style (used when style == "line").
    style : {"line", "dot"}, optional
        "line" for continuous line; "dot" for scatter markers.
    markersize : float, optional
        Marker size in points (used when style == "dot").
    ylim : tuple(float, float), optional
        y-axis limits.
    yticks : sequence of float or None, optional
        y-axis tick locations. If None, leave as default.
    set_ylabel : bool, optional
        If True, set the y-axis label.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the plotted data.
    """
    # 1) Create axis if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # 2) Extract time and magnetic field
    try:
        t = np.asarray(B.time.data)
        Bxyz = np.asarray(B.data)
    except Exception as e:
        raise TypeError("B must be an xarray.DataArray with `.time.data` and `.data`.") from e

    # 3) Compute clock angle
    _, clock = cone_clock_angle(Bxyz)
    series = ts_scalar(t, clock)

    # 4) Plot
    if style == "line":
        plot.plot_line(ax, series, color=color, linewidth=linewidth,
                       linestyle=linestyle, label=label)
    elif style == "dot":
        # matplotlib scatter uses size in points^2
        s = float(markersize) ** 2
        plot.plot_scatter(ax, series, s=s, marker=".", color=color,
                          linewidths=0.0, zorder=10, label=label)
    else:
        raise ValueError("style must be 'line' or 'dot'")

    # 5) Axis cosmetics
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if set_ylabel:
        ax.set_ylabel("Clock angle\n" + r"($^\circ$)")

    # Safe x-limits (handle potential NaNs)
    if t.size > 0:
        tmin = np.nanmin(t)
        tmax = np.nanmax(t)
        if np.isfinite(tmin) and np.isfinite(tmax):
            ax.set_xlim(tmin, tmax)

    return ax

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
                     ylim: Tuple[float, float] = (0.0, 360.0),
                     yticks: Optional[Sequence[float]] = (0, 90, 180, 270, 360),
                     set_ylabel: bool = True) -> plt.Axes:
    """
    Plot the magnetic-field clock angle from an xarray.DataArray time series.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis. If None, a new figure/axis is created.
    B : xarray.DataArray
        Magnetic field time series with shape (N, 3) for (Bx, By, Bz).
        This function explicitly expects:
          - time array at `B.time.data` (length N)
          - field array at `B.data`      (shape (N, 3))
    label : str, optional
        Legend label for the clock-angle line. Default None.
    color : str, optional
        Line color. Default 'black'.
    linewidth : float, optional
        Line width. Default 1.0.
    linestyle : str, optional
        Line style, e.g. '-', '--', ':'. Default '-'.
    ylim : tuple(float, float), optional
        y-axis limits. Default (0.0, 360.0).
    yticks : sequence of float or None, optional
        y-axis tick locations. Default (0, 90, 180, 270, 360).
        Use None to leave ticks unchanged.
    set_ylabel : bool, optional
        If True, set the y-axis label to "Clock angle (°)". Default True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis used for plotting.

    Notes
    -----
    - Clock angle is computed with `py_space_zc.cone_clock_angle(Bxyz)`.
      If it returns (cone, clock), the second output is used.
    - Angles are wrapped to [0, 360) via `np.mod`.

    Examples
    --------
    >>> # B is an xarray.DataArray with B.time.data and B.data (N x 3)
    >>> fig, ax = plt.subplots(figsize=(10, 4))
    >>> plot_clock_angle(ax, B, color="tab:blue", label="Clock angle")
    >>> ax.legend(loc="center right", bbox_to_anchor=(1.1, 0.5), frameon=False)
    >>> plt.show()
    """
    # 1) Create axis if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # 2) Extract time and (Bx, By, Bz) from xarray.DataArray
    try:
        t = np.asarray(B.time.data)
        Bxyz = np.asarray(B.data)
    except Exception as e:
        raise TypeError(
            "B must be an xarray.DataArray with `.time.data` and `.data`."
        ) from e

    Bxyz = np.atleast_2d(Bxyz)

    # 3) Compute clock angle via py_space_zc
    out = cone_clock_angle(Bxyz)
    clock = out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
    clock = np.mod(np.asarray(clock, dtype=float), 360.0)

    # 4) Package as scalar time series and plot
    series = ts_scalar(t, clock)
    try:
        plot.plot_line(ax, series, color=color, linewidth=linewidth,
                       linestyle=linestyle, label=label)
    except Exception:
        # Fallback to raw matplotlib if plot_line expects different input
        ax.plot(t, clock, color=color, linewidth=linewidth,
                linestyle=linestyle, label=label)

    # 5) Axis cosmetics
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if set_ylabel:
        ax.set_ylabel("Clock angle\n" + r"($^\circ$)")
    ax.set_xlim(t[0], t[-1])

    return ax

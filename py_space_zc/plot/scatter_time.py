import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def scatter_time(ax=None, x=None, y=None, time=None,
                 cmap='Spectral_r', size=15,
                 min_nticks=3, start_on_top=False,
                 zorder=None, fontname="Times New Roman", **kwargs):
    """
    Scatter with a time-colored colorbar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If None, create a new figure/axes.
    x, y : array-like
        Coordinates.
    time : array-like of datetime64 or datetime
        Time for each point; assumed in chronological order.
    cmap : str
        Colormap name.
    size : float
        Marker size.
    min_nticks : int
        Minimum number of colorbar ticks (>=3). Start & end included.
    start_on_top : bool
        If True, put the start time at the top of the colorbar.
    zorder : float or None
        Drawing order for the scatter points.
    fontname : str
        Font family used for the colorbar label and tick labels.
    **kwargs
        Additional keyword arguments forwarded to matplotlib.axes.Axes.scatter.

    Returns
    -------
    ax, scatter, cbar
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # ---- Convert time to Matplotlib float dates. ----
    time = np.asarray(time)
    # Support datetime64 and Python datetime inputs.
    if np.issubdtype(time.dtype, np.datetime64):
        time_dt = time.astype('datetime64[s]').astype('O')
    else:
        time_dt = time
    time_num = mdates.date2num(time_dt)

    # ---- Use the start and end times as color limits. ----
    vmin = mdates.date2num(time_dt[0])
    vmax = mdates.date2num(time_dt[-1])

    kwargs.setdefault("edgecolors", "none")
    if zorder is not None:
        kwargs["zorder"] = zorder

    sc = ax.scatter(x, y, c=time_num, s=size, cmap=cmap,
                    vmin=vmin, vmax=vmax, **kwargs)

    # ---- colorbar ----
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.005, aspect=14)

    # Use at least min_nticks ticks and include the start and end times.
    n = max(int(min_nticks), 3)
    ticks = np.linspace(vmin, vmax, n)
    cbar.set_ticks(ticks)

    # Time format.
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    cbar.set_label("Time", rotation=270, labelpad=15, fontname=fontname)
    cbar.ax.yaxis.label.set_fontname(fontname)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname(fontname)

    # Optionally place the start time at the top of the colorbar.
    if start_on_top:
        cbar.ax.invert_yaxis()

    return ax, sc, cbar

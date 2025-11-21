import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


def hist1d(
        ax,
        x,
        y=None,
        bins=50,
        xrange=None,
        xscale='linear',
        stat='count',
        c='C0',
        lw=2,
        label=None,
        marker=None,
        return_all=False,
):
    """
    Draw a 1D binned statistic plot (like hist2d but for 1D).

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw the plot. If None, create a new one.
    x : array-like
        The independent variable (e.g., altitude).
    y : array-like or None
        The dependent variable to apply the statistic to (e.g., density). If None, 'count' is used.
    bins : int or array-like
        Number of bins or bin edges.
    xrange : [float, float], optional
        x-axis range.
    xscale : {'linear', 'log'}, optional
        Axis scale.
    stat : {'count', 'mean', 'median'}, optional
        Statistic to compute in each bin.
    c : str
        Line color.
    lw : float
        Line width.
    label : str or None
        Legend label.
    marker : str or None
        Marker style.
    return_all : bool
        If True, return (ax, bin_centers, stat_values)

    Returns
    -------
    ax : matplotlib.axes.Axes
    bin_centers : ndarray
    stat_values : ndarray
    """

    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)

    if xscale == 'log':
        x = x[x > 0]
        if y is not None:
            y = y[x > 0]

    # Define bin edges
    if isinstance(bins, int):
        if xrange is None:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
        else:
            xmin, xmax = xrange
        if xscale == 'log':
            edges = np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
        else:
            edges = np.linspace(xmin, xmax, bins + 1)
    else:
        edges = np.asarray(bins)

    # Choose statistic
    if stat not in ['count', 'mean', 'median']:
        stat = 'count'

    values = None if stat == 'count' else y
    H, edges, _ = binned_statistic(x, values, statistic=stat, bins=edges)

    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Setup axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(bin_centers, H, c=c, lw=lw, marker=marker, label=label)
    ax.set_xscale(xscale)
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    if return_all:
        return ax, bin_centers, H
    else:
        return ax

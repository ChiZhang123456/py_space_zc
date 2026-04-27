import numpy as np
import matplotlib.pyplot as plt

def bar1d(ax=None, data=None, bins=10, xrange=None, xscale='linear',
          mode='count', color='steelblue', edgecolor='black', alpha=0.6):
    """
    Plot a 1D bar histogram on a given axis with flexible normalization.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new axis is created.
    data : array-like
        Input data array.
    bins : int
        Number of histogram bins.
    xrange : tuple or None
        x-axis limits as (min, max). If None, will be inferred.
    xscale : str
        Scale of x-axis, 'linear' or 'log'.
    mode : str
        Type of histogram: 'count' (default), 'fraction', or 'density'.
        - 'count': raw counts per bin
        - 'fraction': counts normalized by total (i.e., sum = 1)
        - 'density': normalized by bin width (PDF), area under curve = 1
    color : str
        Fill color of bars.
    edgecolor : str
        Edge color of bars.
    alpha : float
        Transparency level.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the histogram.
    bin_centers : ndarray
        Bin center values.
    values : ndarray
        Height of each bar (count/fraction/density).
    """
    if data is None:
        raise ValueError("Input data must be provided.")

    data = np.asarray(data)

    if xscale == 'log':
        data = data[data > 0]

    if xrange is None:
        xrange = (np.min(data), np.max(data))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Compute histogram
    counts, bin_edges = np.histogram(data, bins=bins, range=xrange)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Choose normalization mode
    if mode == 'count':
        values = counts
        ylabel = 'Counts'
    elif mode == 'fraction':
        values = counts / counts.sum()
        ylabel = 'Fraction'
    elif mode == 'density':
        values = counts / (counts.sum() * bin_width)
        ylabel = 'Probability Density'
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'count', 'fraction', or 'density'.")

    # Plot bars
    ax.bar(bin_centers, values, width=bin_width, color=color,
           edgecolor=edgecolor, alpha=alpha)

    ax.set_xscale(xscale)
    ax.set_xlim(xrange)
    ax.set_ylabel(ylabel)

    return ax, bin_centers, values

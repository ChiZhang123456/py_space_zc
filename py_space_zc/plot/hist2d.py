import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d

def hist2d(
    ax,
    x,
    y,
    weight=None,
    bins=50,
    xrange=None,
    yrange=None,
    xscale='linear',
    yscale='linear',
    cscale='linear',
    stat='count',
    contour=False,
    clim=None,
    cmap='Spectral_r',
    contour_levels=10,
    contour_kwargs=None,
    min_count=1,
):
    """
    Draw a 2D histogram using pcolormesh, with optional weighting, masking, and colorbar.

    Parameters
    ----------
    ...
    min_count : int, optional
        Minimum number of samples per bin required. Bins with fewer samples are set to NaN.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the histogram.
    pcm : QuadMesh
        The pcolormesh object.
    cbar : Colorbar
        The colorbar object.
    H : ndarray
        The 2D histogram/statistics result.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    if weight is not None:
        weight = np.asarray(weight)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if xscale == 'log':
        x = x[x > 0]
    if yscale == 'log':
        y = y[y > 0]

    if isinstance(bins, int):
        xbins = ybins = bins
    elif isinstance(bins, (list, tuple, np.ndarray)) and len(bins) == 2:
        xbins, ybins = bins
    else:
        raise ValueError("bins must be int or [xbins, ybins]")

    if isinstance(xbins, int):
        if xrange is None:
            xrange = [np.nanmin(x), np.nanmax(x)]
        if xscale == 'log':
            xedges = np.logspace(np.log10(xrange[0]), np.log10(xrange[1]), xbins + 1)
        else:
            xedges = np.linspace(xrange[0], xrange[1], xbins + 1)
    else:
        xedges = np.asarray(xbins)

    if isinstance(ybins, int):
        if yrange is None:
            yrange = [np.nanmin(y), np.nanmax(y)]
        if yscale == 'log':
            yedges = np.logspace(np.log10(yrange[0]), np.log10(yrange[1]), ybins + 1)
        else:
            yedges = np.linspace(yrange[0], yrange[1], ybins + 1)
    else:
        yedges = np.asarray(ybins)

    # Choose statistic function
    stat_func = stat if stat in ['count', 'sum', 'mean', 'median'] else 'count'

    # Main histogram
    H, xedge, yedge, _ = binned_statistic_2d(
        x, y, weight, statistic=stat_func, bins=[xedges, yedges]
    )

    # Compute bin counts for masking
    Ncount, _, _, _ = binned_statistic_2d(
        x, y, None, statistic='count', bins=[xedges, yedges]
    )

    # Mask bins with too few samples
    H[Ncount < min_count] = np.nan

    # Transpose for pcolormesh
    H = H.T

    # Setup color normalization
    norm = None
    if cscale == 'log':
        H[H <= 0] = np.nan
        if clim is not None:
            norm = LogNorm(vmin=clim[0], vmax=clim[1])
        else:
            norm = LogNorm(vmin=np.nanmin(H), vmax=np.nanmax(H))

    # Plot with pcolormesh
    pcm = ax.pcolormesh(
        xedges, yedges, H,
        cmap=cmap,
        shading='auto',
        norm=norm if cscale == 'log' else None,
        vmin=None if cscale == 'log' else clim[0] if clim else None,
        vmax=None if cscale == 'log' else clim[1] if clim else None
    )

    # Contours
    if contour:
        cs_kwargs = dict(colors='k', linewidths=0.7)
        if contour_kwargs:
            cs_kwargs.update(contour_kwargs)
        Xc, Yc = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
        ax.contour(Xc, Yc, H, levels=contour_levels, **cs_kwargs)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    cbar = plt.colorbar(pcm, ax=ax, pad=0.01, fraction=0.02)

    return ax, pcm, cbar, H

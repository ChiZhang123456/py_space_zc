import matplotlib.pyplot as plt
import numpy as np
from .set_axis import set_axis

def subplot(
        nrows: int,
        ncols: int,
        figsize=(10, 8),
        hspace=0.1,
        wspace=0.1,
        bottom=0.1, top=0.9,
        left=0.1, right=0.9,
        sharex: bool = False,
        sharey: bool = False,
):
    """
    Create a matplotlib figure with subplots using GridSpec.

    Parameters
    ----------
    nrows : int
        Number of rows of subplots (>=1).
    ncols : int
        Number of columns of subplots (>=1).
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default (10, 8).
    hspace, wspace : float, optional
        Vertical / horizontal spacing between subplots.
    bottom, top, left, right : float, optional
        Figure margins in normalized figure coordinates [0, 1].
    sharex, sharey : bool, optional
        If True, subplots share the same x/y-axis. When enabled:
        - Only the bottom row shows x tick labels.
        - Only the leftmost column shows y tick labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs : list[Axes]
        List of axes in row-major order.

    Examples
    --------
    >>> fig, axs = subplot(2, 3, figsize=(9, 6),
    ...                    hspace=0.25, wspace=0.05,
    ...                    sharex=True, sharey=True,
    ...                    bottom=0.08, top=0.95,
    ...                    left=0.1, right=0.95)
    >>> axs[0].plot([1, 2, 3], [1, 4, 9])
    >>> axs[5].scatter([1, 2, 3], [3, 2, 1])
    >>> plt.show()
    """
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be >= 1")

    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize)
    gspec = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        hspace=hspace,
        wspace=wspace,
        bottom=bottom,
        top=top,
        left=left,
        right=right,
    )

    axs = []
    for i in range(nrows):
        for j in range(ncols):
            if i == 0 and j == 0:
                ax = fig.add_subplot(gspec[i, j])
            else:
                ax = fig.add_subplot(
                    gspec[i, j],
                    sharex=axs[j] if sharex else None,          # same column
                    sharey=axs[i * ncols] if sharey else None,  # same row
                )
            axs.append(ax)

    # If sharing, hide redundant tick labels
    if sharex or sharey:
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                show_x = (i == nrows - 1) or (not sharex)
                show_y = (j == 0) or (not sharey)
                try:
                    set_axis(axs[idx],
                             show_xticklabels=show_x,
                             show_yticklabels=show_y)
                except Exception:
                    axs[idx].tick_params(labelbottom=show_x,
                                         labelleft=show_y)

    return fig, axs

import matplotlib.pyplot as plt
from .set_axis import set_axis

def subplot(
        nrows: int,
        ncols: int,
        fig=None,
        figsize=(10, 8),
        hspace=0.1,
        wspace=0.1,
        bottom=0.1, top=0.9,
        left=0.1, right=0.9,
        sharex: bool = False,
        sharey: bool = False,
):
    """
    Create subplots on an existing figure (if provided) or a new one.

    Parameters
    ----------
    nrows : int
        Number of rows of subplots (>=1).
    ncols : int
        Number of columns of subplots (>=1).
    fig : matplotlib.figure.Figure or None, optional
        Target figure. If None, a new figure will be created.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Used only when `fig is None`.
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
        The figure (existing or newly created).
    axs : list[matplotlib.axes.Axes]
        Axes in row-major order.
    """

    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be >= 1")

    # Use existing figure or create a new one
    if fig is None:
        fig = plt.figure(figsize=figsize)

    # Build a GridSpec on this figure
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

    # Hide redundant tick labels when sharing
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

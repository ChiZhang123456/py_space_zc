import matplotlib.pyplot as plt
import numpy as np
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
    Create subplots with a 2D axes array return and global Times New Roman font.

    Parameters
    ----------
    nrows : int
        Number of rows of subplots (>=1).
    ncols : int
        Number of columns of subplots (>=1).
    fig : matplotlib.figure.Figure or None, optional
        Target figure. If None, a new figure will be created.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Used only if `fig is None`.
    hspace, wspace : float, optional
        Vertical and horizontal spacing between subplots.
    bottom, top, left, right : float, optional
        Figure margins in normalized figure coordinates [0, 1].
    sharex, sharey : bool, optional
        If True, subplots share the same x or y-axis.
        - sharex: Only the bottom row displays x-tick labels.
        - sharey: Only the leftmost column displays y-tick labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : numpy.ndarray
        A 2D array of Axes objects of shape (nrows, ncols).
        Access via axs[row, col].
    """

    # --- 1. Global Font Configuration ---
    # Set default font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # Ensure minus signs display correctly with Times New Roman
    plt.rcParams['axes.unicode_minus'] = False

    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be at least 1.")

    # --- 2. Initialize Figure ---
    if fig is None:
        fig = plt.figure(figsize=figsize)

    # --- 3. Define GridSpec Layout ---
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

    # --- 4. Create Subplots in a 2D Structure ---
    # We use an object-type numpy array to allow 2D indexing: axs[i, j]
    axs = np.empty((nrows, ncols), dtype=object)

    for i in range(nrows):
        for j in range(ncols):
            # Axis sharing logic:
            # sharex links to the first row (row 0)
            # sharey links to the first column (col 0)
            target_sharex = axs[0, j] if (sharex and i > 0) else None
            target_sharey = axs[i, 0] if (sharey and j > 0) else None

            ax = fig.add_subplot(
                gspec[i, j],
                sharex=target_sharex,
                sharey=target_sharey
            )
            axs[i, j] = ax

    # --- 5. Manage Tick Label Visibility ---
    if sharex or sharey:
        for i in range(nrows):
            for j in range(ncols):
                # Visibility conditions:
                # Show X labels only if it's the bottom row or sharing is off
                show_x = (i == nrows - 1) or (not sharex)
                # Show Y labels only if it's the first column or sharing is off
                show_y = (j == 0) or (not sharey)

                try:
                    # Attempt to use custom helper function
                    set_axis(axs[i, j], show_xticklabels=show_x, show_yticklabels=show_y)
                except Exception:
                    # Fallback to native Matplotlib parameters
                    axs[i, j].tick_params(labelbottom=show_x, labelleft=show_y)

    return fig, axs
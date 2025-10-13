import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

def adjust_colorbar(ax, cbar_or_cax, pad=0.01, height_ratio=1.0, width=0.01):
    """
    Adjust the colorbar position relative to an axis.
    Works whether you pass a Colorbar object or its Axes (cbar.ax).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis that the colorbar aligns to.
    cbar_or_cax : matplotlib.colorbar.Colorbar or matplotlib.axes.Axes
        Either the Colorbar object, or its Axes (cbar.ax).
    pad : float, optional
        Horizontal spacing from the axis (in figure normalized coords).
    height_ratio : float, optional
        Fraction of the axis height that the colorbar should occupy (0 < r <= 1).
    width : float, optional
        Colorbar width (in figure normalized coords).

    Returns
    -------
    cax : matplotlib.axes.Axes
        The Axes hosting the colorbar (useful for further tweaks).
    """
    if not isinstance(ax, Axes):
        raise TypeError("`ax` must be a matplotlib.axes.Axes.")

    # Resolve cax from either Colorbar or Axes
    if isinstance(cbar_or_cax, Colorbar):
        cax = cbar_or_cax.ax
    elif isinstance(cbar_or_cax, Axes):
        cax = cbar_or_cax
    elif hasattr(cbar_or_cax, "ax") and isinstance(getattr(cbar_or_cax, "ax"), Axes):
        cax = getattr(cbar_or_cax, "ax")
    else:
        raise TypeError("`cbar_or_cax` must be a Colorbar or an Axes (cbar.ax).")

    # Basic parameter checks
    if not (0 < height_ratio <= 1):
        raise ValueError("`height_ratio` must be in (0, 1].")
    if width <= 0:
        raise ValueError("`width` must be > 0.")
    if pad < 0:
        raise ValueError("`pad` must be >= 0.")

    fig = ax.figure

    # Try to finalize layout before reading positions (helps with tight/constrained layout)
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception:
        pass

    # Reference axis position
    pos = ax.get_position()  # Bbox in figure normalized coords

    # Compute vertically centered colorbar box
    cbar_height = pos.height * height_ratio
    y0 = pos.y0 + 0.5 * (pos.height - cbar_height)
    x0 = pos.x1 + pad

    cax.set_position([x0, y0, width, cbar_height])
    return cax

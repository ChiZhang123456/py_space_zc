import matplotlib.pyplot as plt
from typing import Optional

def add_colorbar(
    ax,
    pcm,
    orientation: str = "vertical",   # "vertical" or "horizontal"
    where: Optional[str] = None,     # vertical: "right"/"left"; horizontal: "top"/"bottom"
    size_ratio: float = 0.5,         # fraction of the axis length along the colorbar
    thickness_ratio: float = 0.06,   # fraction of the axis size perpendicular to the colorbar
    pad: float = 0.01,               # gap between axis and colorbar, in figure coords
    **cbar_kwargs
):
    """
    Add a colorbar positioned relative to an axis, using axis-relative sizing.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    pcm : ScalarMappable
        Mappable (e.g., result of pcolormesh/imshow).
    orientation : {"vertical","horizontal"}
        Colorbar orientation.
    where : str or None
        For vertical: "right" (default) or "left".
        For horizontal: "top" (default) or "bottom".
    size_ratio : float
        Length of the colorbar as a fraction of the axis length along the bar
        (height for vertical, width for horizontal).
    thickness_ratio : float
        Thickness of the colorbar as a fraction of the axis size perpendicular
        to the bar (width for vertical, height for horizontal).
    pad : float
        Gap between the axis and colorbar in figure coordinates (0–1).
    **cbar_kwargs :
        Extra kwargs forwarded to `fig.colorbar`.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar.
    """
    fig = ax.figure
    pos = ax.get_position()  # Bbox in figure coordinates

    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")

    if where is None:
        where = "right" if orientation == "vertical" else "top"

    if orientation == "vertical":
        # Size along bar = fraction of axis height; thickness = fraction of axis width
        height_cbar = size_ratio * pos.height
        width_cbar  = thickness_ratio * pos.width
        y0 = pos.y0 + 0.5 * (pos.height - height_cbar)

        if where == "right":
            x0 = pos.x1 + pad
        elif where == "left":
            x0 = pos.x0 - pad - width_cbar
        else:
            raise ValueError("For vertical, where must be 'right' or 'left'.")

        cax = fig.add_axes([x0, y0, width_cbar, height_cbar])
        cbar = fig.colorbar(pcm, cax=cax, orientation="vertical", **cbar_kwargs)

    else:  # horizontal
        # Size along bar = fraction of axis width; thickness = fraction of axis height
        width_cbar  = size_ratio * pos.width
        height_cbar = thickness_ratio * pos.height
        x0 = pos.x0 + 0.5 * (pos.width - width_cbar)

        if where == "top":
            y0 = pos.y1 + pad
        elif where == "bottom":
            y0 = pos.y0 - pad - height_cbar
        else:
            raise ValueError("For horizontal, where must be 'top' or 'bottom'.")

        cax = fig.add_axes([x0, y0, width_cbar, height_cbar])
        cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal", **cbar_kwargs)

    return cbar

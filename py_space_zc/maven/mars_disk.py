from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D


def plot_mars(
    ax=None,
    center=(0.0, 0.0),
    radius=1.0,
    texture=True,
    texture_path=None,
    alpha=1.0,
    zorder=10,
    rotate_deg=0.0,
    preserve_limits=True,
    facecolor="black",
    edgecolor="none",
    lw=1.0,
    **imshow_kwargs,
):
    """
    Plot Mars as a radius-1 disk on a 2-D projection axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Target axis. If None, a new axis is created.
    center : tuple, default (0, 0)
        Disk center in data coordinates.
    radius : float, default 1
        Disk radius in data coordinates, typically 1 Rm.
    texture : bool, default True
        If True, use the bundled Mars true-color disk image. If False, draw a
        solid disk.
    texture_path : str or path-like, optional
        Custom image path. The image should preferably have a transparent
        background.
    alpha : float, default 1
        Disk opacity.
    zorder : float, default 10
        Matplotlib z-order.
    rotate_deg : float, default 0
        Counterclockwise image rotation angle in degrees.
    preserve_limits : bool, default True
        Restore axis limits after drawing.
    facecolor, edgecolor, lw
        Styling used when texture=False.
    **imshow_kwargs
        Extra keyword arguments forwarded to ax.imshow.
    """
    if ax is None:
        _, ax = plt.subplots()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx, cy = center

    if texture:
        if texture_path is None:
            texture_path = Path(__file__).with_name("mars_globe_true_color.png")
        img = plt.imread(texture_path)
        extent = (cx - radius, cx + radius, cy - radius, cy + radius)
        defaults = {
            "origin": "upper",
            "interpolation": "bilinear",
        }
        defaults.update(imshow_kwargs)
        im = ax.imshow(
            img,
            extent=extent,
            alpha=alpha,
            zorder=zorder,
            **defaults,
        )
        if rotate_deg:
            tr = Affine2D().rotate_deg_around(cx, cy, rotate_deg) + ax.transData
            im.set_transform(tr)
    else:
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.fill(
            cx + radius * np.cos(theta),
            cy + radius * np.sin(theta),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )

    if preserve_limits:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return ax

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from py_space_zc import maven, tianwen_1, plot
import matplotlib.patches as patches


def plot_maven_tianwen_yz(
        ax,
        tint,
        cmap='Spectral_r',
        size=15,
        equal_aspect=True,
        sat_offset=(0.0, 0.0),     # (dx, dy) offset for start-point markers
        cbar_pad=0.04,             # padding between colorbar and main axis
        cbar_fraction=0.02,        # relative width of the colorbar
        cbar_aspect=10,            # aspect ratio of the colorbar (length/width)
    ):
    """
    Plot MAVEN & Tianwen-1 spacecraft trajectories in Y-Z (MSO) coordinates,
    using a shared colorbar for time (formatted as HH:MM:SS) and markers
    at the starting positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis. If None, a new figure/axis will be created.
    tint : [str, str]
        Time interval, e.g. ["2021-11-16T08:40:00", "2021-11-16T09:20:00"].
    cmap : str
        Colormap shared by both spacecraft trajectories.
    size : float
        Marker size for the trajectory scatters.
    equal_aspect : bool
        If True, enforce equal aspect ratio on the axes.
    sat_offset : tuple(float, float)
        Offset applied to the start-point markers (in data coordinates).
    cbar_pad : float
        Spacing between the colorbar and the axes (relative units).
    cbar_fraction : float
        Width of the colorbar relative to the axes.
    cbar_aspect : float
        Aspect ratio of the colorbar (length/width).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Main axis with trajectories plotted.
    cbar : matplotlib.colorbar.Colorbar
        Shared colorbar representing time (HH:MM:SS).
    leg : matplotlib.legend.Legend
        Legend showing MAVEN and Tianwen-1.
    """

    # --------------------------------------------------------------
    # Load trajectory data
    B   = maven.get_data(tint, 'B')
    Btw = tianwen_1.get_data(tint, 'B')

    # Positions in Mars radii (Rm = 3390 km)
    Pmvn = B["Pmso"].data  / 3390.0
    Ptw  = Btw["Pmso"].data / 3390.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # --------------------------------------------------------------
    # Scatter plots with time as colormap
    t_mvn = B["Pmso"].time.data
    t_tw  = Btw["Pmso"].time.data

    # Use helper function scatter_time (remove its default colorbars)
    _, s1, c1 = plot.scatter_time(ax, Pmvn[:, 1], Pmvn[:, 2], t_mvn,
                                  cmap=cmap, size=size, min_nticks=3)
    _, s2, c2 = plot.scatter_time(ax, Ptw[:, 1], Ptw[:, 2],  t_tw,
                                  cmap=cmap, size=size, min_nticks=3)
    if c1 is not None: c1.remove()
    if c2 is not None: c2.remove()

    # --------------------------------------------------------------
    # Add start-point markers for both spacecraft
    ax.scatter(Pmvn[0, 1] + sat_offset[0], Pmvn[0, 2] + sat_offset[1],
               marker="*", s=5*size, c='black',
               edgecolors="black", linewidths=1.2, zorder=10)
    ax.scatter(Ptw[0, 1] + sat_offset[0], Ptw[0, 2] + sat_offset[1],
               marker="*", s=5*size, c='violet',
               edgecolors="violet", linewidths=1.2, zorder=10)

    # --------------------------------------------------------------
    # Axis formatting
    if equal_aspect:
        ax.set_aspect('equal')

    ax.set_xlabel(r"$Y_{\mathrm{MSO}}$ (R$_\mathrm{M}$)")
    ax.set_ylabel(r"$Z_{\mathrm{MSO}}$ (R$_\mathrm{M}$)")

    ax.add_patch(patches.Circle((0, 0), radius=1, color='black', alpha=0.35, zorder=10))


    # --------------------------------------------------------------
    # Shared colorbar (time in HH:MM:SS)
    cbar = plt.colorbar(s1, ax=ax, fraction=cbar_fraction, pad=cbar_pad, aspect=cbar_aspect)
    pos = ax.get_position()
    # cbar.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.01, pos.height - 0.02])

    cbar.set_label("Time")

    # Format colorbar ticks as HH:MM:SS
    cbar.ax.yaxis_date()
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # --------------------------------------------------------------
    # Legend with custom markers
    handles = [
        Line2D([0], [0], marker="*", color="black",  linestyle="None",
               label="MAVEN",     markersize=12, markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="violet", linestyle="None",
               label="Tianwen-1", markersize=12, markeredgecolor="violet"),
    ]
    leg = ax.legend(handles=handles, loc="best", frameon=False)

    return ax, cbar, leg


# Example usage
if __name__ == "__main__":
    tint = ["2021-11-16T08:40:00", "2021-11-16T09:20:00"]
    ax, cbar, leg = plot_maven_tianwen_yz(
        ax=None, tint=tint, cmap='Spectral_r',
        size=15, cbar_pad=0.06, cbar_fraction=0.05, cbar_aspect=18
    )
    pos = ax.get_position()
    cbar.ax.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.015, pos.height - 0.02])
    plt.show()

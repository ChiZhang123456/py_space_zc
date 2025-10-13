import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from matplotlib.colors import Normalize
from typing import Iterable, Tuple, Optional
from py_space_zc import maven, tianwen_1, plot, lmn
import matplotlib.patches as patches


def plot_maven_tianwen_yz_mse(
    ax,
    tint: Iterable[str],
    xmse: np.ndarray = np.array((1.0, 0.0, 0.0)),
    ymse: np.ndarray = np.array((0.0, 1.0, 0.0)),
    zmse: np.ndarray = np.array((0.0, 0.0, 1.0)),
    cmap: str = "Spectral_r",
    size: float = 15.0,
    equal_aspect: bool = True,
    sat_offset: Tuple[float, float] = (0.0, 0.0),
    cbar_pad: float = 0.04,
    cbar_fraction: float = 0.02,
    cbar_aspect: float = 10.0,
    mars_face_alpha: float = 0.35,
    mars_edge: bool = False,
):
    """
    Plot MAVEN & Tianwen-1 trajectories in the Y–Z MSE plane with a shared
    time colorbar and start-point markers.

    The function fetches spacecraft positions with `py_space_zc.maven.get_data`
    and `py_space_zc.tianwen_1.get_data` for the provided interval `tint`,
    converts positions from MSO to MSE using supplied basis vectors, and
    overlays two time-colored scatters (shared colormap). A single colorbar
    is drawn and formatted as HH:MM:SS.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis. If None, a new figure/axis is created.
    tint : [str, str]
        Time interval, e.g., ["2021-11-16T08:40:00", "2021-11-16T09:20:00"].
        Must be two ISO-like timestamp strings understood by your data loaders.
    xmse, ymse, zmse : np.ndarray, shape (3,)
        Orthonormal basis vectors defining the MSE frame (x̂,ŷ,ẑ).
        Defaults are canonical axes. If you already have a proper MSE frame,
        pass those here (e.g., from IMF + cross products).
    cmap : str
        Colormap used for both spacecraft.
    size : float
        Marker size for the scatters.
    equal_aspect : bool
        If True, set axis aspect to 'equal'.
    sat_offset : (float, float)
        Offset applied to the *start* marker of each spacecraft in data units.
        Useful to avoid overlapping with the first scatter marker.
    cbar_pad : float
        Padding between colorbar and axes.
    cbar_fraction : float
        Relative width of the colorbar.
    cbar_aspect : float
        Aspect ratio (length/width) of the colorbar.
    mars_face_alpha : float
        Alpha for the filled Mars disk (radius = 1 R_M).
    mars_edge : bool
        If True, draw an edge around the Mars disk for clearer boundary.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plot.
    cbar : matplotlib.colorbar.Colorbar or None
        The shared time colorbar. None if creation failed.
    leg : matplotlib.legend.Legend
        Legend with star markers for MAVEN and Tianwen-1.

    Notes
    -----
    - Positions are assumed to be provided in km and converted to Mars radii
      using R_M = 3390 km. Adjust if your upstream returns different units.
    - `plot.scatter_time` is assumed to return (ax, PathCollection, colorbar)
      and to map times to a Matplotlib date-like scalar. We suppress its
      individual colorbars to draw a shared one.
    """
    # -------------------------------
    # Input & axis preparation
    # -------------------------------
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if not (isinstance(tint, (list, tuple)) and len(tint) == 2):
        raise ValueError("tint must be a 2-element list/tuple of [t_start, t_end]")

    # -------------------------------
    # Data loading
    # -------------------------------
    B_mvn = maven.get_data(tint, "B")
    B_tw  = tianwen_1.get_data(tint, "B")

    # Convert km → R_M and MSO → MSE (keep only Y,Z columns for plotting)
    R_M = 3390.0  # km
    Pmvn_mso = B_mvn["Pmso"].data / R_M
    Ptw_mso  = B_tw["Pmso"].data  / R_M

    # lmn() here is used as a general linear transform into the (x,y,z) basis (MSE)
    Pmvn = lmn(Pmvn_mso, xmse, ymse, zmse)
    Ptw  = lmn(Ptw_mso,  xmse, ymse, zmse)

    # -------------------------------
    # Time arrays (used for coloring)
    # -------------------------------
    t_mvn = B_mvn["Pmso"].time.data
    t_tw  = B_tw["Pmso"].time.data

    if len(Pmvn) == 0 or len(Ptw) == 0:
        raise ValueError("Empty trajectory after loading/filtering; nothing to plot.")

    # -------------------------------
    # Scatter with time colormap
    # We deliberately remove default colorbars to build a shared one.
    # -------------------------------
    _, s1, c1 = plot.scatter_time(
        ax, Pmvn[:, 1], Pmvn[:, 2], t_mvn,
        cmap=cmap, size=size, min_nticks=3
    )
    _, s2, c2 = plot.scatter_time(
        ax, Ptw[:, 1],  Ptw[:, 2],  t_tw,
        cmap=cmap, size=size, min_nticks=3
    )

    c1.remove()
    c2.remove()

    # Align both scatters to the same normalization (important if helper returns different norms)
    if hasattr(s1, "norm") and hasattr(s2, "norm"):
        shared_norm = Normalize(vmin=min(s1.norm.vmin, s2.norm.vmin),
                                vmax=max(s1.norm.vmax, s2.norm.vmax))
        s1.set_norm(shared_norm)
        s2.set_norm(shared_norm)

    # -------------------------------
    # Start-point markers (stars)
    # -------------------------------
    ax.scatter(
        Pmvn[0, 1] + sat_offset[0],
        Pmvn[0, 2] + sat_offset[1],
        marker="*", s=5 * size, c="black",
        edgecolors="black", linewidths=1.2, zorder=10
    )
    ax.scatter(
        Ptw[0, 1] + sat_offset[0],
        Ptw[0, 2] + sat_offset[1],
        marker="*", s=5 * size, c="violet",
        edgecolors="violet", linewidths=1.2, zorder=10
    )

    # -------------------------------
    # Axes cosmetics
    # -------------------------------
    if equal_aspect:
        ax.set_aspect("equal")

    ax.set_xlabel(r"$Y_{\mathrm{MSE}}$ (R$_\mathrm{M}$)")
    ax.set_ylabel(r"$Z_{\mathrm{MSE}}$ (R$_\mathrm{M}$)")

    # Mars disk at origin (radius = 1 R_M)
    facecolor = "black" if not mars_edge else "none"
    circle = patches.Circle(
        (0, 0), radius=1,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.0 if mars_edge else 0.0,
        alpha=mars_face_alpha,
        zorder=9
    )
    ax.add_patch(circle)

    # -------------------------------
    # Shared colorbar (time axis)
    # -------------------------------
    try:
        cbar = plt.colorbar(
            s1, ax=ax,
            fraction=cbar_fraction,
            pad=cbar_pad,
            aspect=cbar_aspect
        )
        cbar.set_label("Time (UTC)")

        # Format colorbar ticks as time. We use AutoDateLocator + DateFormatter
        # on the colorbar axis to ensure robust formatting across intervals.
        cax = cbar.ax
        cax.yaxis.set_major_locator(mdates.AutoDateLocator())
        cax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        # If scatter_time encoded times as floats not recognized as dates,
        # uncomment the next line to suppress misleading minor ticks:
        # cax.yaxis.set_minor_formatter(NullFormatter())
    except Exception:
        # If colorbar creation fails (e.g., helper returns unsupported mappables),
        # keep plotting without a colorbar rather than raising.
        cbar = None

    # -------------------------------
    # Legend (custom handles)
    # -------------------------------
    handles = [
        Line2D([0], [0], marker="*", color="black",  linestyle="None",
               label="MAVEN",     markersize=12, markeredgecolor="black"),
        Line2D([0], [0], marker="*", color="violet", linestyle="None",
               label="Tianwen-1", markersize=12, markeredgecolor="violet"),
    ]
    leg = ax.legend(handles=handles, loc="best", frameon=False)

    return ax, cbar, leg


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    tint = ["2021-11-16T08:40:00", "2021-11-16T09:20:00"]
    ax, cbar, leg = plot_maven_tianwen_yz_mse(
        ax=None, tint=tint, cmap="Spectral_r",
        size=15, cbar_pad=0.06, cbar_fraction=0.05, cbar_aspect=18,
        mars_edge=True  # set True to outline Mars disk
    )
    # Optional: manually adjust colorbar position after layout
    if cbar is not None:
        pos = ax.get_position()
        cbar.ax.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.015, pos.height - 0.02])

    plt.show()

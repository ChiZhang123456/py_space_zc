import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch, Rectangle


# ===== Function definition =====
def connect_axes(ax1, ax2,
                 color='k', lw=1.2, ls='--',
                 alpha=1.0, draw_rect=False,
                 global_side='bottom', zoom_side='top'):
    """
    Connect two time-series axes by drawing linking lines and an optional
    rectangle indicating the zoomed region.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        The upper (global overview) axis.

    ax2 : matplotlib.axes.Axes
        The lower (zoomed-in) axis. Its x-limits define the zoom interval.

    color : str, optional
        Color of the connection lines and rectangle (default: 'k').

    lw : float, optional
        Line width (default: 1.2).

    ls : str, optional
        Line style (default: '--').

    alpha : float, optional
        Transparency level (default: 1.0).

    draw_rect : bool, optional
        Whether to draw a rectangle on the global axis (default: False).

    global_side : {'bottom', 'top'}, optional
        Side of the global axis from which the lines originate.

    zoom_side : {'top', 'bottom'}, optional
        Side of the zoom axis where the lines terminate.
    """

    # Get x-limits from the zoomed axis (in matplotlib date format)
    x0, x1 = ax2.get_xlim()

    # Determine y-coordinates on the global axis
    y1_min, y1_max = ax1.get_ylim()
    yA = y1_min if global_side == 'bottom' else y1_max

    # Determine y-coordinates on the zoom axis
    y2_min, y2_max = ax2.get_ylim()
    yB = y2_max if zoom_side == 'top' else y2_min

    # Optionally draw a rectangle highlighting the zoom region
    if draw_rect:
        rect = Rectangle(
            (x0, y1_min),
            x1 - x0,
            y1_max - y1_min,
            linewidth=lw,
            edgecolor=color,
            facecolor='none',
            linestyle=ls,
            alpha=alpha,
            zorder=5
        )
        ax1.add_patch(rect)

    # Draw left connection line
    con1 = ConnectionPatch(
        xyA=(x0, yA), coordsA=ax1.transData,
        xyB=(x0, yB), coordsB=ax2.transData,
        color=color, lw=lw, ls=ls, alpha=alpha
    )

    # Draw right connection line
    con2 = ConnectionPatch(
        xyA=(x1, yA), coordsA=ax1.transData,
        xyB=(x1, yB), coordsB=ax2.transData,
        color=color, lw=lw, ls=ls, alpha=alpha
    )

    # Add connection lines to the figure
    fig = ax1.figure
    fig.add_artist(con1)
    fig.add_artist(con2)

    return con1, con2


# ===== Main test block =====
if __name__ == "__main__":

    # Generate a synthetic time series (mock data)
    t = pd.date_range("2025-06-15 12:00:00", periods=500, freq="s")
    y = np.sin(np.linspace(0, 20, len(t))) + 0.2 * np.random.randn(len(t))

    # Create figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # ---- Global panel ----
    axs[0].plot(t, y)
    axs[0].set_title("Global")

    # ---- Zoomed-in panel ----
    t_start = np.datetime64("2025-06-15T12:03:00")
    t_end   = np.datetime64("2025-06-15T12:03:20")

    mask = (t >= t_start) & (t <= t_end)

    axs[1].plot(t[mask], y[mask])
    axs[1].set_xlim(t_start, t_end)
    axs[1].set_title("Zoom")

    # ---- Connect the two panels ----
    connect_axes(axs[0], axs[1])

    # Format the x-axis
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.tight_layout()
    plt.show()
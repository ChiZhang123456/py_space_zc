from py_space_zc import maven, tianwen_1, plot, delta_angle, cone_clock_angle, ts_scalar
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

def plot_maven_tianwen_dca(
        ax: Optional[plt.Axes],
        tint,
        option: str = "line",          # "line" or "dot"
        linewidth: float = 1.5,
        markersize: float = 4.0,
        color: str = "C0",
        ylim: Tuple[float, float] = (-180.0, 180.0),
        yticks: Optional[Sequence[float]] = (0, 45, 90, 135, ),
        set_ylabel: bool = True,
        add_zero: bool = True,
) -> plt.Axes:
    """
    Plot Δ clock angle = CA(MAVEN) - CA(Tianwen-1), mapped to [-180, 180] deg.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis. If None, a new figure/axis will be created.
    tint : [str, str]
        Time interval.
    option : {"line", "dot"}
        "line" -> line plot; "dot" -> scatter markers.
    linewidth : float
        Line width (used when option == "line").
    markersize : float
        Marker size in points (used when option == "dot").
    color : str
        Line/marker color.
    ylim : (float, float)
        y-axis limits.
    yticks : sequence of float or None
        y-axis ticks; None to keep defaults.
    set_ylabel : bool
        If True, set the y-axis label.
    add_zero : bool
        If True, draw a horizontal 0° reference line.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the plotted Δ clock angle.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Fetch data
    B   = maven.get_data(tint, 'B')
    Btw = tianwen_1.get_data(tint, 'B')

    # Compute clock angles and wrap into time series
    _, ca_mvn = cone_clock_angle(B["Bmso"].data)
    _, ca_tw  = cone_clock_angle(Btw["Bmso"].data)
    ca_mvn_ts = ts_scalar(B["Bmso"].time.data, ca_mvn)
    ca_tw_ts  = ts_scalar(Btw["Bmso"].time.data, ca_tw)

    # Δ clock angle with resampling/alignment handled internally
    dca = delta_angle(ca_mvn_ts, ca_tw_ts)  # returns ts_scalar(..., diff in [-180,180])

    # Plot
    if option == "line":
        plot.plot_line(ax, dca, color=color, linestyle="-", linewidth=linewidth, label=r"$\Delta$CA")
    elif option == "dot":
        s = float(markersize) ** 2  # matplotlib scatter uses pt^2
        plot.plot_scatter(ax, dca, s=s, marker=".", color=color, linewidths=0.0, zorder=10, label=r"$\Delta$CA")
    else:
        raise ValueError("option must be 'line' or 'dot'")

    # Cosmetics
    if add_zero:
        ax.axhline(0.0, color="0.7", lw=1.0, ls="--", zorder=0)

    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if set_ylabel:
        ax.set_ylabel(r"$\Delta \phi$ ($^\circ$)")
    ax.set_ylim(0, 180)
    return ax


if __name__ == "__main__":
    tint = ["2021-12-27T18:05","2021-12-27T18:38"]
    plot_maven_tianwen_dca(None, tint, 'dot',markersize = 5)
    plt.show()
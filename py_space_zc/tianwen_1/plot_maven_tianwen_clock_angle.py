from py_space_zc import maven, tianwen_1, plot
import matplotlib.pyplot as plt

def plot_maven_tianwen_clock_angle(
        ax,
        tint,
        option: str = "line",
        linewidth: float = 1.5,
        markersize: float = 4.0,
):
    """
    Plot clock angle comparison between MAVEN and Tianwen-1.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis will be created.
    tint : [str, str]
        Time interval.
    option : {"line", "dot"}, optional
        Plot style:
        - "line": use line plot
        - "dot" : use scatter/dots
    linewidth : float, optional
        Line width (used if option="line").
    markersize : float, optional
        Marker size (used if option="dot").

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the plotted clock angles.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # fetch data
    B   = maven.get_data(tint, 'B')
    Btw = tianwen_1.get_data(tint, 'B')

    # choose style
    if option == "line":
        plot.plot_clock_angle(ax, B["Bmso"], color="tab:red",
                              label="MAVEN", linestyle="-",
                              linewidth=linewidth)
        plot.plot_clock_angle(ax, Btw["Bmso"], color="tab:blue",
                              label="Tianwen-1", linestyle="-",
                              linewidth=linewidth)

    elif option == "dot":
        plot.plot_clock_angle(ax, B["Bmso"], color="tab:red",
                              label="MAVEN", style="dot")
        plot.plot_clock_angle(ax, Btw["Bmso"], color="tab:blue",
                              label="Tianwen-1", style="dot")

    else:
        raise ValueError("option must be 'line' or 'dot'")

    ax.set_ylabel(r"$\phi$ (°)")
    # ax.legend()

    return ax

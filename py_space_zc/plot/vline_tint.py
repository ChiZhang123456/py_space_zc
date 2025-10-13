from matplotlib import dates

def vline_tint(axs, tint, ymin: float = 0, ymax: float = 1,
               color="k", linestyle="--", linewidth=1.0, **kwargs):
    r"""Add two vertical lines across the time Axes.

    Notes
    -----
    - `tint` must be two ISO-8601 strings: ["YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DDTHH:MM:SS"].
    - Lines are drawn at tint[0] and tint[1], spanning from `ymin` to `ymax` in axis units.

    Parameters
    ----------
    axs : list of matplotlib.axes.Axes
        Axes to draw lines on.
    tint : list[str]
        Time interval [start, stop], ISO-8601 strings.
    ymin, ymax : float
        Vertical span in axis units (0–1).
    color, linestyle, linewidth :
        Default line style (can be overridden by **kwargs).
    **kwargs :
        Extra args forwarded to `Axes.axvline`.

    Returns
    -------
    list
        The same list of Axes.
    """
    t0 = dates.datestr2num(tint[0])
    t1 = dates.datestr2num(tint[1])
    if t0 > t1:
        t0, t1 = t1, t0

    style = dict(color=color, linestyle=linestyle, linewidth=linewidth)
    style.update(kwargs)

    for ax in axs:
        ax.axvline(t0, ymin=ymin, ymax=ymax, **style)
        ax.axvline(t1, ymin=ymin, ymax=ymax, **style)

    return axs

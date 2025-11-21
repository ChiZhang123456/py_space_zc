import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import FixedLocator

def set_xtick(ax, tint, n=5, fmt='%H:%M:%S', fontsize=11):
    """
    Set xticks for a matplotlib axes with formatted datetime labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to set xticks on.

    tint : list of str
        Time range as a list of two ISO 8601 strings,
        e.g., ['2022-06-19T12:21:00', '2022-06-19T12:27:00'].

    n : int, optional
        Number of xticks to generate. Default is 5.

    fmt : str, optional
        Datetime formatting string for xtick labels.
        Examples: '%H:%M:%S', '%H:%M', '%M:%S', '%.2f' (for float-based formats).
        Default is '%H:%M:%S'.

    fontsize : int, optional
        Font size of xtick labels. Default is 13.
    """

    # --- Convert tint strings to datetime objects
    t1 = datetime.strptime(tint[0], "%Y-%m-%dT%H:%M:%S")
    t2 = datetime.strptime(tint[1], "%Y-%m-%dT%H:%M:%S")

    # --- Convert to Matplotlib's internal float date format
    t1_num = mdates.date2num(t1)
    t2_num = mdates.date2num(t2)

    # --- Generate evenly spaced ticks
    xticks = np.linspace(t1_num, t2_num, n)

    # --- Apply settings to the axis
    ax.set_xlim(t1, t2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [mdates.num2date(t).strftime(fmt) for t in xticks],
        fontsize=fontsize
    )

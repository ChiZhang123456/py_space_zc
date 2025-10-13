import numpy as np
import matplotlib.pyplot as plt

def add_time_title(ax, tint, format: str = "yyyy/mm/dd HH:MM - HH:MM", **kwargs):
    """
    Add a formatted title to a matplotlib axis based on a time interval or a single timestamp.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis to which the title will be added.
    tint : np.datetime64, datetime.datetime, or list/tuple of two such values
        A single time (ts) or a time interval [ts, te].
    format : str
        Format string with placeholders:
            - "yyyy" : 4-digit year
            - "mm"   : 2-digit month
            - "dd"   : 2-digit day
            - "HH"   : 2-digit hour (24h)
            - "MM"   : 2-digit minute
            - "SS"   : 2-digit second

        Examples:
            - "yyyy/mm/dd HH:MM - HH:MM"
            - "yyyy/mm/dd HH:MM:SS"
            - "Start: yyyy-mm-dd HH:MM"

    **kwargs :
        Additional keyword arguments passed to `ax.set_title`.

    Returns
    -------
    None
    """
    # Convert time(s) to datetime.datetime
    if isinstance(tint, (np.datetime64, np.datetime64.__base__)):
        ts = np.datetime64(tint).astype("datetime64[ms]").item()
        te = None
    elif isinstance(tint, (list, tuple)) and len(tint) == 2:
        ts = np.datetime64(tint[0]).astype("datetime64[ms]").item()
        te = np.datetime64(tint[1]).astype("datetime64[ms]").item()
    else:
        raise ValueError("tint must be a single time or a list/tuple of two times.")

    # Placeholder mapping
    fmt_map = {
        "yyyy": "%Y",
        "mm": "%m",
        "dd": "%d",
        "HH": "%H",
        "MM": "%M",
        "SS": "%S"
    }

    def convert_format(fmt):
        """Replace custom placeholders with strftime-compatible format codes."""
        out = fmt
        for k, v in fmt_map.items():
            out = out.replace(k, v)
        return out

    # Format string handling
    if te is not None and " - " in format:
        left_fmt, right_fmt = format.split(" - ", 1)
    else:
        left_fmt, right_fmt = format, None

    left_fmt = convert_format(left_fmt)
    if right_fmt:
        right_fmt = convert_format(right_fmt)

    # Construct title string
    if te is not None and right_fmt:
        title = ts.strftime(left_fmt) + " - " + te.strftime(right_fmt)
    else:
        title = ts.strftime(left_fmt)

    # Apply title
    ax.set_title(title, **kwargs)


# ==================== Example Usage ====================
if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2)

    # Case 1: time interval
    t1 = np.datetime64("2023-09-17T05:27:00")
    t2 = np.datetime64("2023-09-17T05:35:00")
    add_time_title(axs[0], [t1, t2], "yyyy/mm/dd HH:MM - HH:MM", color="blue", fontsize=12)

    # Case 2: single timestamp
    t3 = np.datetime64("2023-09-17T06:00:00")
    add_time_title(axs[1], t3, "yyyy-mm-dd HH:MM:SS", color="green", fontsize=12)

    plt.tight_layout()
    plt.show()

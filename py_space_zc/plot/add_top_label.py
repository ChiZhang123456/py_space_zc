import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import dates
from typing import Tuple, Union
import numpy as np
from py_space_zc.plot import span_tint

def to_mpl_num(t):
    """Convert time to matplotlib float date format."""
    if isinstance(t, str):
        return dates.datestr2num(t)
    elif np.issubdtype(type(t), np.datetime64):
        return dates.date2num(t.astype('O'))
    elif hasattr(t, 'timestamp'):  # datetime.datetime
        return dates.date2num(t)
    else:
        raise TypeError(f"Unsupported time type: {type(t)}")

def add_top_axes(ax: Axes,
                 height_ratio: float = 0.1,
                 spacing: float = 0.01) -> Axes:
    """Create a small axes directly above the input axis, sharing the same x-axis."""
    fig = ax.figure
    pos = ax.get_position()

    new_height = pos.height * height_ratio
    new_bottom = pos.y1 + spacing

    ax_top = fig.add_axes([
        pos.x0,
        new_bottom,
        pos.width,
        new_height
    ], sharex=ax)

    # Hide ticks without modifying the shared XAxis object
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_top.set_yticks([])
    ax_top.set_facecolor("none")
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    return ax_top



def add_top_label(ax: Axes,
                  tint: Tuple[Union[str, float], Union[np.datetime64]],
                  color: str = 'red',
                  alpha: float = 0.5,
                  height_ratio: float = 0.05,
                  spacing: float = 0.001,
                  **kwargs) -> Axes:
    """Add a highlight bar above an axis to mark a time interval."""
    ax_top = add_top_axes(ax, height_ratio=height_ratio, spacing=spacing)

    t_start, t_stop = [to_mpl_num(t) for t in tint]
    ax_top.axvspan(t_start, t_stop, color=color, alpha=alpha, **kwargs)

    return ax_top


if __name__ == '__main__':
    from py_space_zc import maven
    from pyrfu.pyrf import extend_tint

    tint = ["2022-02-22T11:45:00", "2022-02-22T12:10:00"]
    tint_shad = extend_tint(tint, [100, -30])

    fig, ax = plt.subplots()
    maven.plot_B(ax, tint)  # 使用传入的 ax

    add_top_label(ax, tint = tint_shad, color="red", alpha=0.3)

    plt.show()

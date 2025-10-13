import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def scatter_time(ax=None, x=None, y=None, time=None,
                 cmap='Spectral_r', size=15,
                 min_nticks=3, start_on_top=False):
    """
    Scatter with a time-colored colorbar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If None, create a new figure/axes.
    x, y : array-like
        Coordinates.
    time : array-like of datetime64 or datetime
        Time for each point; assumed in chronological order.
    cmap : str
        Colormap name.
    size : float
        Marker size.
    min_nticks : int
        Minimum number of colorbar ticks (>=3). Start & end included.
    start_on_top : bool
        If True, put the start time at the top of the colorbar.

    Returns
    -------
    ax, scatter, cbar
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # ---- 时间 -> Matplotlib 浮点日期 ----
    time = np.asarray(time)
    # 支持 datetime64 和 Python datetime
    if np.issubdtype(time.dtype, np.datetime64):
        time_dt = time.astype('datetime64[s]').astype('O')  # 转 python datetime（秒精度）
    else:
        time_dt = time
    time_num = mdates.date2num(time_dt)

    # ---- 设定 vmin/vmax 为起止时间，去掉上下空白 ----
    vmin = mdates.date2num(time_dt[0])
    vmax = mdates.date2num(time_dt[-1])

    sc = ax.scatter(x, y, c=time_num, s=size, cmap=cmap,
                    edgecolors='none', vmin=vmin, vmax=vmax)

    # ---- colorbar ----
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.005, aspect=14)

    # 至少 min_nticks 个刻度，并强制包含起点与终点
    n = max(int(min_nticks), 3)
    ticks = np.linspace(vmin, vmax, n)
    cbar.set_ticks(ticks)

    # 时间格式
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    cbar.set_label("Time", rotation=270, labelpad=15)

    # 起始时间放顶部（可选）
    if start_on_top:
        cbar.ax.invert_yaxis()

    return ax, sc, cbar

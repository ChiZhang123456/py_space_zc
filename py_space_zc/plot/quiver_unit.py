import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def quiver_unit(ax, time, U, V, *,
                y0=0.0,
                length_y=0.5,
                pivot='tail',
                scale=None,          # 若提供，则覆盖 length_y
                **kwargs):
    """
    在时间轴上绘制单位化箭头（方向来自 U,V，长度统一）。
    - angles='uv'：方向只看(U,V)，不受坐标缩放影响
    - scale_units='y'：用 y 轴数据单位控制长度
    - length_y：每个箭头在 y 轴上的可视长度
    - 如果传入 scale，则直接用该 scale（等价于 length_y = 1/scale）
    """
    t_num = mdates.date2num(time)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    N = np.sqrt(U**2 + V**2)
    N[N == 0] = 1.0
    Uu, Vu = U/N, V/N

    if np.isscalar(y0):
        Y0 = np.zeros_like(t_num, dtype=float) + float(y0)
    else:
        Y0 = np.asarray(y0, dtype=float)

    # 计算 scale：优先使用用户给的 scale，否则用 length_y 的倒数
    _scale = float(scale) if scale is not None else 1.0/float(length_y)

    # 防止用户在 kwargs 里无意覆盖关键参数
    kwargs.pop('angles', None)
    kwargs.pop('scale_units', None)
    kwargs.pop('scale', None)

    q = ax.quiver(t_num, Y0, Uu, Vu,
                  angles='uv',
                  scale_units='y',
                  scale=_scale,
                  pivot=pivot,
                  **kwargs)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_yticklabels("")
    ax.set_xlim(time[0], time[-1])
    
    return q

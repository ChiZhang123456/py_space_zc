import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def quiver_unit(ax, time, U, V, *,
                y0=0.0,
                length_y=0.5,
                pivot='tail',
                scale=None,          # Override length_y when provided.
                **kwargs):
    """
    Plot normalized arrows on a time axis.

    The arrow direction is set by U and V, while the displayed length is fixed.
    `angles='uv'` makes the direction depend only on U and V, independent of
    axis scaling. `scale_units='y'` controls the length in y-axis data units.
    If `scale` is provided, it is used directly, equivalent to
    `length_y = 1 / scale`.
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

    # Prefer user-provided scale, otherwise use the inverse of length_y.
    _scale = float(scale) if scale is not None else 1.0/float(length_y)

    # Prevent accidental override of key quiver parameters.
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

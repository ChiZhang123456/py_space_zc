import numpy as np
import xarray as xr
from py_space_zc import ts_vec_xyz

def background_B(
    Bwave: xr.DataArray,
    window_length: float,
    overlap: float,
    include_partial: bool = True,
    time_center: str = "midpoint",
):
    """
    Estimate the background (mean) magnetic field by averaging over sliding time windows.

    Parameters
    ----------
    Bwave : xarray.DataArray
        Magnetic field time series with dimensions (time, 3).
        - Bwave.time.data : np.ndarray of np.datetime64 (monotonically increasing).
        - Bwave.data      : np.ndarray of shape (N, 3) containing (Bx, By, Bz).
    window_length : float
        Length of each averaging window in seconds. For example, 10.0 → 10-second windows.
    overlap : float
        Amount of overlap between consecutive windows in seconds.
        The step between successive window starts is: step = window_length - overlap.
        Example: window_length=10 s, overlap=5 s → start a new 10 s window every 5 s.
    include_partial : bool, optional (default=True)
        If True, the last window is included even if it is shorter than window_length,
        as long as it contains at least one sample. If False, the trailing partial
        window is discarded.
    time_center : {"midpoint", "mean"}, optional (default="midpoint")
        Method to define the representative time of each averaged window:
        - "midpoint": halfway between window start and window end.
        - "mean": average of all sample timestamps included in the window.

    Returns
    -------
    B_bg : xarray.DataArray
        Time series of the background (averaged) magnetic field with shape (time, 3).
        Each row corresponds to the average field vector within one window.
        Returned as a `ts_vec_xyz` DataArray, with coordinates:
        - time : np.ndarray of np.datetime64 window times.
        - comp : ["x", "y", "z"] vector components.

    Notes
    -----
    - NaN values are ignored via np.nanmean. If all samples of a component in
      a window are NaN, that component will return NaN for that window.
    - Works with both regularly and irregularly sampled time series
      (averaging is performed by actual time, not sample index).
    - Requires strictly increasing Bwave.time values.
    """
    # ---- Basic parameter checks --------------------------------------------
    if window_length <= 0:
        raise ValueError("window_length must be > 0.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    step = window_length - overlap
    if step <= 0:
        raise ValueError("Require step = window_length - overlap > 0.")

    # ---- Extract time and data ---------------------------------------------
    t = Bwave.time.data     # np.datetime64 array
    B = Bwave.data          # (N, 3)

    # Convert absolute time to relative seconds from t0
    t0 = t[0].astype("datetime64[ns]")
    t_ns = t.astype("datetime64[ns]")
    time_s = (t_ns - t0).astype("timedelta64[ns]").astype(np.int64) * 1e-9
    t_start, t_end = time_s[0], time_s[-1]

    # ---- Build list of window start times ----------------------------------
    starts = []
    s = t_start
    while s <= t_end:   # include windows that begin before last sample
        starts.append(s)
        s += step

    # ---- Average within each window ----------------------------------------
    out_times, out_vals = [], []
    wlen = window_length

    for s in starts:
        e = s + wlen
        if (not include_partial) and (e > t_end):
            # skip trailing partial window if not allowed
            break

        # mask samples inside the window
        mask = (time_s >= s) & (time_s < e) if e <= t_end else (time_s >= s)
        if not np.any(mask):
            continue  # no samples in this window

        # average over the window (ignore NaNs)
        B_win = B[mask, :]
        B_avg = np.nanmean(B_win, axis=0)

        # representative time
        if time_center == "midpoint":
            rep_sec = 0.5 * (s + min(e, t_end))
        elif time_center == "mean":
            rep_sec = float(np.nanmean(time_s[mask]))
        else:
            raise ValueError("time_center must be 'midpoint' or 'mean'.")

        # convert back to datetime64
        rep_ns = int(np.round(rep_sec * 1e9))
        rep_time = (t0 + np.timedelta64(rep_ns, "ns")).astype("datetime64[ns]")

        out_times.append(rep_time)
        out_vals.append(B_avg)

    # ---- Construct output DataArray ----------------------------------------
    out_times = np.array(out_times, dtype="datetime64[ns]")
    out_vals = np.vstack(out_vals)
    B_bg = ts_vec_xyz(out_times, out_vals)

    return B_bg


if __name__ == "__main__":
    from py_space_zc import maven, plot, ts_scalar
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pyrfu import pyrf

    tint = ["2022-01-24T08:06:30", "2022-01-24T08:09:00"]
    B_high, B = maven.load_data(tint, ['B_high','B'])
    window_length = 1.0
    overlap_length = 0.5
    Bbgd = background_B(B_high['Bmso'], window_length, overlap_length)
    fig, ax = plot.subplot(3,1, sharex=True)
    plot.plot_line(ax[0], B_high['Bmso'])
    plot.plot_line(ax[1], B['Bmso'])
    plot.plot_line(ax[2], Bbgd)
    ax[0].set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    ax[1].set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    ax[2].set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    plt.show()

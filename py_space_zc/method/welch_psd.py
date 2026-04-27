import numpy as np
from scipy import signal
from py_space_zc import lmn, ts_spectr, ts_vec_xyz


def psd_welch(Bwave, nperseg=1024 * 64):
    """
    Bottom-level function: Calculates PSD for a single Bwave segment.
    Automatically calculates fs, handles NaNs, and transforms to FAC coordinates.
    """
    data = Bwave.data.copy()
    time_raw = Bwave.time.data

    # 1. Handle NaNs (Linear interpolation)
    # FFT-based methods like Welch require continuous data without NaNs
    if np.isnan(data).any():
        for i in range(3):  # Loop through X, Y, Z components
            mask = np.isnan(data[:, i])
            if mask.any():
                # Use valid indices to interpolate NaN values
                data[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask, i])

    # 2. Automatically calculate sampling frequency (fs)
    # Convert np.datetime64 to float seconds
    time_sec = (time_raw - time_raw[0]).astype('float64') / 1e9
    dt = np.median(np.diff(time_sec))
    fs = 1.0 / dt

    # 3. Calculate background magnetic field and center time
    Bbgd_vec = np.nanmean(data, axis=0)
    Bbgd_mag = np.linalg.norm(Bbgd_vec)
    avg_time = time_raw[len(time_raw) // 2]

    # 4. Construct FAC (LMN) coordinate system
    # L: Parallel to the background field
    L = Bbgd_vec / Bbgd_mag
    # M: Perpendicular to L and MSO-X [1,0,0]
    M = np.cross(L, np.array([1, 0, 0]))
    M /= np.linalg.norm(M)
    # N: Completes the right-handed system
    N = np.cross(L, M)
    N /= np.linalg.norm(N)

    # 5. Transform to FAC and Detrend (subtract mean)
    B_fac = lmn(data, L, M, N)
    dB_para = B_fac[:, 0] - np.nanmean(B_fac[:, 0])
    dB_p1 = B_fac[:, 1] - np.nanmean(B_fac[:, 1])
    dB_p2 = B_fac[:, 2] - np.nanmean(B_fac[:, 2])

    # 6. Perform Welch PSD Analysis
    n_actual = min(nperseg, len(dB_para))
    noverlap = int(n_actual / 2)
    f, p_para = signal.welch(dB_para, fs=fs,
                             nperseg=n_actual,
                             noverlap=noverlap,
                             detrend='constant',
                             average='median')
    _, p_p1 = signal.welch(dB_p1, fs=fs, nperseg=n_actual,
                           noverlap=noverlap,
                           detrend='constant',
                           average='median')
    _, p_p2 = signal.welch(dB_p2, fs=fs, nperseg=n_actual,
                           noverlap=noverlap,
                           detrend='constant',
                           average='median')

    return {
        'f': f,
        'time': avg_time,
        'Bbgd': Bbgd_vec,
        'psd_para': p_para,
        'psd_perp': p_p1 + p_p2
    }


def psd_welch_sliding(Bwave, window_length,
                      overlap_ratio=0.5, nperseg=1024 * 64):
    """
    Top-level function: Manages sliding windows and wraps results into ts_ objects.
    """
    # Calculate fs once for window indexing
    time_raw = Bwave.time.data
    time_sec = (time_raw - time_raw[0]).astype('float64') / 1e9
    fs = 1.0 / np.median(np.diff(time_sec))

    pts_per_window = int(window_length * fs)
    step = int(pts_per_window * (1 - overlap_ratio))

    tmp_time = []
    tmp_Bbgd = []
    tmp_psd_para = []
    tmp_psd_perp = []
    f_axis = None

    # Iterate through data using sliding windows
    for start_idx in range(0, len(Bwave.data) - pts_per_window + 1, step):
        end_idx = start_idx + pts_per_window

        # Slice the DataArray for the current window
        B_segment = Bwave.isel(time=slice(start_idx, end_idx))

        # Call the bottom-level function
        out = psd_welch(B_segment, nperseg=nperseg)

        if f_axis is None:
            f_axis = out['f']

        tmp_time.append(out['time'])
        tmp_Bbgd.append(out['Bbgd'])
        tmp_psd_para.append(out['psd_para'])
        tmp_psd_perp.append(out['psd_perp'])

    # Wrap results into py_space_zc specific objects
    res_time = np.array(tmp_time)

    results = {
        'Bbgd': ts_vec_xyz(res_time, np.array(tmp_Bbgd)),
        'psd_para': ts_spectr(res_time, f_axis, np.array(tmp_psd_para), comp_name='f'),
        'psd_perp': ts_spectr(res_time, f_axis, np.array(tmp_psd_perp), comp_name='f')
    }

    return results
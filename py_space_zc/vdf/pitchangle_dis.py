import numpy as np

def pitchangle_dis(data, pa_dis, delta_angles = 22.5):
    """
    Create the pitch angle distribution (PAD).

    Parameters
    ----------
    data : ndarray
        Input array of shape (ntime, nenergy, ndirection) or
        (ntime, nenergy, nphi, ntheta), containing plasma quantities
        like flux or phase space density.

    pa_dis : ndarray
        Pitch angle values corresponding to each energy, directional bin,
        same shape as `data`.

    delta_angles : float
        Width of each pitch angle bin in degrees.

    Returns
    -------
    pad_arr : ndarray
        Pitch angle distribution array with shape (ntime, nenergy, npad),
        where `npad` = number of pitch angle bins (typically 180 / delta_angles).
    """
    # Define pitch angle bin centers and widths
    angles_v = np.linspace(delta_angles, 180, int(180 / delta_angles))  # e.g., [10, 20, ..., 180]
    d_angles = np.median(np.diff(angles_v)) * np.ones(len(angles_v))    # Bin width
    pitch_angles = angles_v - d_angles / 2                              # Bin centers

    n_angles = len(angles_v)

    # Create empty array to hold intermediate PAD slices per bin
    dists = np.empty((n_angles,) + data.shape)  # Shape: (npad, ntime, nenergy, ...)

    for i in range(n_angles):
        angle_min = angles_v[i] - d_angles[i]
        angle_max = angles_v[i]

        # Mask values that fall outside the current pitch angle bin
        mask_lower = pa_dis < angle_min
        mask_upper = pa_dis > angle_max

        # Copy data and mask out-of-bin values with NaN
        dists[i] = np.copy(data)
        dists[i][mask_lower | mask_upper] = np.nan

    # For 3D input: (ntime, nenergy, ndirection)
    if data.ndim == 3:
        # Average over direction axis (axis=2 → third dimension)
        pad_arr = np.nanmean(dists, axis=3)
        # Transpose to (ntime, nenergy, npad)
        pad_arr = np.transpose(pad_arr, (1, 2, 0))

    # For 4D input: (ntime, nenergy, nphi, ntheta)
    elif data.ndim == 4:  #for 4D data, use the sum method
        # Sum over direction axes (nphi and ntheta)
        pad_arr = np.nansum(dists, axis=(3, 4))
        # Transpose to (ntime, nenergy, npad)
        pad_arr = np.transpose(pad_arr, (1, 2, 0))

    else:
        raise ValueError("Input data must be either 3D or 4D array.")

    return pitch_angles, pad_arr

if __name__ == '__main__':
    from py_space_zc import maven
    tint = ["2022-10-19T00:54:00", "2022-10-19T00:59:00"]
    swea_pad = maven.load_data(tint,'swea_pad')
    theta, pad = pitchangle_dis(swea_pad['DEF'], swea_pad['PA'], delta_angles=22.5)
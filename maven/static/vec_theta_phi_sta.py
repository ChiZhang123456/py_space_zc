import numpy as np
from .mso2sta import mso2sta
from py_space_zc import cart2sph, ts_scalar, ts_vec_xyz

def vec_theta_phi_sta(inp, d1):
    """
    Convert a vector time series from MSO (Mars-Solar-Orbital) coordinates
    to STATIC instrument coordinates, then compute the polar angle (theta)
    and azimuth (phi) in the STATIC frame for each time step.

    This function also supports a time-invariant input vector, e.g. [1, 0, 0].
    In that case, the vector is assumed to be constant in the MSO frame and is
    tiled across d1['time'], then wrapped into a TSeries via ts_vec_xyz.

    Parameters
    ----------
    inp :
        - TSeries-like (vector, shape = [n_time, 3]) with attributes:
            .time : array-like of datetime64 or equivalent
            .data : ndarray of shape (n_time, 3), columns = [X, Y, Z] in MSO
        - OR array-like of shape (3,) representing a time-invariant MSO vector
          (e.g., [1, 0, 0]).
    d1 : dict
        STATIC D1 dataset dictionary (e.g., from py_space_zc.maven.get_data).
        Must contain:
            - 'time'    : array-like, target time grid for output
            - 'sta2mso' : ndarray of shape (n_time, 3, 3),
                          rotation matrices from STATIC -> MSO coordinates

    Returns
    -------
    theta_ts : TSeries-like (scalar)
        Time series of theta (polar angle from +Z axis in STATIC frame) in degrees.
    phi_ts : TSeries-like (scalar)
        Time series of phi (azimuth angle from +X axis in STATIC frame) in degrees.

    Notes
    -----
    - If `inp` is a static 3-vector, it is tiled to len(d1['time']) and converted
      to a vector TSeries in MSO using ts_vec_xyz(d1['time'], ...).
    - Frame transformation from MSO to STATIC uses mso2sta(), which applies the
      inverse of STA->MSO rotation matrices supplied in d1['sta2mso'].
    - cart2sph(x, y, z) is expected to return (r, theta, phi) with:
         theta = polar angle (0° = +Z axis, 90° = XY plane),
         phi   = azimuth   (0° = +X axis, 90° = +Y axis).

    Examples
    --------
    1) TSeries input:
        >>> theta_ts, phi_ts = vec_theta_phi_sta(vec_mso_ts, d1)

    2) Static vector input:
        >>> theta_ts, phi_ts = vec_theta_phi_sta([1, 0, 0], d1)
        # Interpreted as a constant MSO vector, tiled across d1['time'].
    """
    # -- Step 0: Normalize input into a vector TSeries in MSO frame --
    # Case A: already a TSeries-like with .time and .data
    if hasattr(inp, "data") and hasattr(inp, "time"):
        vec_mso = inp
    else:
        # Case B: a static 3-vector -> tile across d1['time'] and wrap with ts_vec_xyz
        arr = np.asarray(inp)
        if arr.ndim != 1 or arr.size != 3:
            raise ValueError(
                "When `inp` is array-like, it must be a 1D 3-vector, e.g., [1, 0, 0]."
            )
        n_time = len(d1["time"])
        # Create a tiled (n_time, 3) array without modifying the original arr
        data_tiled = np.tile(arr.reshape(1, 3), (n_time, 1))
        vec_mso = ts_vec_xyz(d1["time"], data_tiled)
        # Optional metadata for clarity
        if getattr(vec_mso, "attrs", None) is not None:
            vec_mso.attrs["Coordinates"] = "MSO"

    # -- Step 1: Transform the vector from MSO to STATIC coordinates --
    inp_sta = mso2sta(vec_mso, d1)  # aligned to d1['time']

    # -- Step 2: Extract STATIC-frame components (X, Y, Z) --
    x = inp_sta.data[:, 0]
    y = inp_sta.data[:, 1]
    z = inp_sta.data[:, 2]

    # -- Step 3: Convert to spherical coordinates (r, theta, phi) --
    #   theta: polar angle from +Z (deg), phi: azimuth from +X (deg)
    _, theta, phi = cart2sph(x, y, z)

    # -- Step 4: Package outputs as scalar TSeries aligned to d1['time'] --
    theta_ts = ts_scalar(d1["time"], theta)
    phi_ts   = ts_scalar(d1["time"], phi)

    # Attach coordinate metadata for downstream clarity
    if getattr(theta_ts, "attrs", None) is not None:
        theta_ts.attrs["Coordinates"] = "STATIC"
    if getattr(phi_ts, "attrs", None) is not None:
        phi_ts.attrs["Coordinates"] = "STATIC"

    return theta_ts, phi_ts

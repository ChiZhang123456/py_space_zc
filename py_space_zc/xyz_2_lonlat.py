import numpy as np

def xyz_2_lonlat(Pmso):
    """
    Convert Cartesian coordinates (X, Y, Z) to (lat, lon, altitude).

    Parameters
    ----------
    Pmso : array_like
        Shape (3,), (1,3), or (N,3). Position(s) in meters.
    Rm : float
        Mars radius in meters. Default is 3390e3.

    Returns
    -------
    lat : ndarray
        Latitude(s) in degrees.
    lon : ndarray
        Longitude(s) in degrees (0–360).
    h : ndarray
        Altitude(s) above Mars radius in meters.
    """
    Pmso = np.asarray(Pmso)

    # Ensure shape (N, 3)
    if Pmso.ndim == 1 and Pmso.shape[0] == 3:
        Pmso = Pmso.reshape(1, 3)
    elif Pmso.ndim == 2 and Pmso.shape[1] != 3:
        raise ValueError("Input must have shape (N, 3) or (3,)")

    x, y, z = Pmso[:, 0], Pmso[:, 1], Pmso[:, 2]
    r = np.linalg.norm(Pmso, axis=1)

    lat = np.arcsin(z / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    lon = np.where(lon < 0, lon + 360, lon)
    return lat, lon, r

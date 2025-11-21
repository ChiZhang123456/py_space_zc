import numpy as np

def lonlat_2_xyz(lat, lon, r):
    """
    Convert (latitude, longitude, radius) to Cartesian coordinates (X, Y, Z).

    Parameters
    ----------
    lat : float or array_like
        Latitude(s) in degrees.
    lon : float or array_like
        Longitude(s) in degrees.
    r : float or array_like
        Radius from Mars center [m].

    Returns
    -------
    xyz : ndarray
        Cartesian coordinates (X, Y, Z), shape (N, 3) or (3,) if scalar input.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    r = np.asarray(r)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    xyz = np.stack((x, y, z), axis=-1)
    return xyz

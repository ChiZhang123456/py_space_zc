# -*- coding: utf-8 -*-
"""
Coordinate Transformation Utilities

This module provides functions for converting between Cartesian, spherical,
and polar coordinate systems. All functions support scalar or NumPy array inputs.

Functions
---------
- cart2sph : Convert (x, y, z) → (r, theta, phi)
- sph2cart : Convert (r, theta, phi) → (x, y, z)
- cart2pol : Convert (x, y) → (r, theta)
- pol2cart : Convert (r, theta) → (x, y)

Author: Chi Zhang
Email: zhangchi9508@gmail.com
"""

import numpy as np

from typing import Tuple, Union

ArrayLike = Union[float, int, np.ndarray]

def cart2sph(
    x: ArrayLike,
    y: ArrayLike = None,
    z: ArrayLike = None,
    *,
    degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates (x, y, z) to spherical.

    Parameters
    ----------
    x, y, z : float | array-like | None
        Cartesian coordinates. If `x` has shape (N, 3) and `y`, `z` are None,
        it is interpreted as stacked XYZ.
    output :
        - "r-theta-phi": returns (r, theta, phi)
            theta is colatitude (inclination) ∈ [-90, 90] deg or [-π/2, π/2] rad
            phi is azimuth ∈ [0, 360) deg or [0, 2π) rad
    degrees : bool, default True
        If True, angular outputs are in degrees; otherwise in radians.

    Returns
    -------
    (a, b, c) : tuple of np.ndarray
        According to `output`.

    Notes
    -----
    - This function does not modify its inputs (no in-place ops on inputs).
    - r = 0 points are mapped to lon=0, lat=0 (or theta=π/2), which is a common, benign convention.

    Examples
    --------
    >>> cart2sph(1, 1, 1)  # default: (r, lon, lat) in degrees
    (array([1.73205081]), array([45.]), array([35.26438968]))

    >>> pts = np.array([[1, 0, 0], [0, 1, 0]])
    >>> cart2sph(pts)  # (r, lon, lat)
    (array([1., 1.]), array([0., 90.]), array([0., 0.]))

    >>> cart2sph(1, 1, 1, output="r-theta-phi")  # (r, colatitude, azimuth)
    (array([1.73205081]), array([54.73561032]), array([45.]))
    """
    arr = np.asarray(x)

    # Accept (N,3) packed input
    if y is None and z is None and arr.ndim == 2 and arr.shape[-1] == 3:
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    # Radius
    r = np.sqrt(x * x + y * y + z * z)

    # Azimuth in [-pi, pi] then normalize to [0, 2pi)
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2.0 * np.pi)

    # Colatitude theta in [0, pi], safe divide for r=0
    # where r==0 -> z/r treated as 0 so theta = arccos(0) = pi/2
    cos_theta = np.divide(z, r, out=np.zeros_like(r, dtype=float), where=r > 0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    theta = np.arccos(cos_theta)
    lat = (np.pi / 2.0) - theta

    if degrees:
        return r, np.degrees(lat), np.degrees(phi)
    else:
        return r, lat, phi



def sph2cart(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z).
    Supports scalar, (N,) arrays, or (N, 3) arrays (as separate r, theta, phi).

    Parameters
    ----------
    r, theta, phi : float or array-like
        Spherical coordinates.

    Returns
    -------
    x, y, z : ndarray
        Cartesian coordinates.

    Example
    -------
    >>> sph2cart(1, np.pi/2, np.pi)
    (-1.0, 0.0, 0.0)
    >>> r = [1, 1]
    >>> theta = [np.pi/2, np.pi/2]
    >>> phi = [0, np.pi/2]
    >>> sph2cart(r, theta, phi)
    (array([1., 0.]), array([0., 1.]), array([0., 0.]))
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cart2pol(x, y=None):
    """
    Convert 2D Cartesian coordinates (x, y) to polar (r, theta).
    Supports scalar, (N,) arrays, or (N, 2) arrays.

    Parameters
    ----------
    x, y : float, array-like, or None
        Cartesian coordinates. If x is (N, 2), y can be None.

    Returns
    -------
    r, theta : ndarray
        r     : Radius
        theta : Angle in radians [-π, π]

    Example
    -------
    >>> cart2pol(1, 1)
    (1.414, 0.785)
    >>> pts = np.array([[1, 0], [0, 1]])
    >>> cart2pol(pts)
    (array([1., 1.]), array([0., 1.5708]))
    """
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 2:
        x, y = arr[:, 0], arr[:, 1]
    else:
        x, y = np.asarray(x), np.asarray(y)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def pol2cart(r, theta):
    """
    Convert polar coordinates (r, theta) to Cartesian (x, y).
    Supports scalar or (N,) arrays.

    Parameters
    ----------
    r : float or array-like
        Radius
    theta : float or array-like
        Angle in radians

    Returns
    -------
    x, y : ndarray
        Cartesian coordinates

    Example
    -------
    >>> pol2cart(1, np.pi/2)
    (0.0, 1.0)
    >>> r = [1, 2]
    >>> theta = [0, np.pi/2]
    >>> pol2cart(r, theta)
    (array([1., 0.]), array([0., 2.]))
    """
    r = np.asarray(r)
    theta = np.asarray(theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

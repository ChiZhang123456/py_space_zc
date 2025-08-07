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

def cart2sph(x, y=None, z=None):
    """
    Convert Cartesian coordinates (x, y, z) to spherical (r, theta, phi).
    Supports scalar, (N,) arrays, or (N, 3) arrays.

    Parameters
    ----------
    x, y, z : float, array-like, or None
        Cartesian coordinates.
        If x is of shape (N, 3), y and z can be None.

    Returns
    -------
    r, theta, phi : ndarray
        r     : Radius
        theta : Polar angle (inclination) in radians [0, π]
        phi   : Azimuthal angle in radians [-π, π]

    Example
    -------
    >>> cart2sph(1, 1, 1)
    (1.732, 0.955, 0.785)
    >>> pts = np.array([[1, 0, 0], [0, 1, 0]])
    >>> cart2sph(pts)
    (array([1., 1.]), array([1.5708, 1.5708]), array([0., 1.5708]))
    """
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 3:
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.zeros_like(r), where=r != 0))
    phi = np.arctan2(y, x)
    return r, theta, phi


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

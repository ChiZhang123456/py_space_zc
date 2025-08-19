"""
Author: Chi Zhang

Description:
This module provides functions to:
- Compute the L2 norm of vectors or matrices.
- Normalize vectors or matrices using L2 normalization.
- Compute cross products for 1D or 2D arrays.

Functions:
- norm(x): Compute the L2 norm of a 1D vector, or the row-wise L2 norm of a 2D array.
- normalize(x): Normalize a 1D vector or each row of a 2D array to unit length.
- cross(a, b): Compute the 3D cross product of two 1D vectors, or row-wise cross product for two 2D arrays.

Usage Examples:
>>> norm([3, 4])
5.0

>>> norm([[3, 4, 0], [0, 0, 5]])
array([[5.],
       [5.]])

>>> normalize([3, 4])
array([0.6, 0.8])

>>> normalize([[3, 4, 0], [0, 0, 5]])
array([[0.6, 0.8, 0. ],
       [0. , 0. , 1. ]])
"""

import numpy as np

def norm(x):
    """
    Compute the L2 norm of a 1D array, or row-wise L2 norms for a 2D array.

    Parameters
    ----------
    x : array-like
        Input array. Can be 1D or 2D.

    Returns
    -------
    float or ndarray
        - If input is 1D: returns a scalar representing the L2 norm.
        - If input is 2D: returns an array of shape (n_rows, 1) with the L2 norm of each row.

    Raises
    ------
    ValueError
        If input is not 1D or 2D.
    """
    x = np.array(x, dtype=float)

    if x.ndim == 1:
        return np.linalg.norm(x)

    elif x.ndim == 2:
        return np.linalg.norm(x, axis=1, keepdims=True)

    else:
        raise ValueError("Only 1D or 2D arrays are supported.")

def normalize(x):
    """
    Perform L2 normalization on a 1D or 2D array.

    Parameters
    ----------
    x : array-like
        Input array. Can be 1D or 2D.

    Returns
    -------
    ndarray
        - If input is 1D: returns a unit-length vector with the same shape.
        - If input is 2D: returns an array of the same shape where each row is normalized to unit length.
        - If the norm of any vector is zero, that vector is returned as a zero vector.

    Raises
    ------
    ValueError
        If input is not 1D or 2D.
    """
    x = np.array(x, dtype=float)

    if x.ndim == 1:
        norm_val = np.linalg.norm(x)
        return x / norm_val if norm_val != 0 else np.zeros_like(x)

    elif x.ndim == 2:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return x / norms

    else:
        raise ValueError("Only 1D or 2D arrays are supported.")



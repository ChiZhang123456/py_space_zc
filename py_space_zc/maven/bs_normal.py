import numpy as np


def bs_normal(Pmso):
    """
    Calculate the normal vector to the Mars bow shock at given position(s) in units of Mars radius (Rm).

    Parameters
    ----------
    Pmso : ndarray
        Input position(s), shape should be (3,), (1,3), or (n,3), in units of Mars radius (Rm).

    Returns
    -------
    normal : ndarray
        Normalized bow shock normal vector(s), same shape as P. Entries will be NaN where input is invalid.
    """
    # Ensure P is a 2D array
    Pmso = np.atleast_2d(Pmso)
    if Pmso.shape[1] != 3:
        raise ValueError("Input must have shape (3,), (1,3), or (n,3)")

    # Constants
    e = 1.026
    L = 2.081
    xF = 0.6

    X = Pmso[:, 0]
    Y = Pmso[:, 1]
    Z = Pmso[:, 2]

    # Calculate model-predicted R (cylindrical) and observed R
    Rcal = np.sqrt((e ** 2 - 1) * (X - xF) ** 2 - 2 * L * e * (X - xF) + L ** 2)
    Robs = np.sqrt(Y ** 2 + Z ** 2)

    # Gradients of the bow shock function
    dFdX = 2 * (e ** 2 - 1) * (X - xF) - 2 * e * L
    dFdY = -2 * Y
    dFdZ = -2 * Z

    # Normal vectors
    normals = np.stack((dFdX, dFdY, dFdZ), axis=1)

    # Normalize the normal vectors
    norm_magnitude = np.linalg.norm(normals, axis=1, keepdims=True)
    normal = normals / norm_magnitude

    # Ensure normal x-component is positive
    flip = normal[:, 0] <= 0
    normal[flip] *= -1

    # Invalidate normals where Rcal is negative or deviates too far from Robs
    invalid = (Rcal < 0) | (np.abs(Rcal - Robs) > 0.2)
    normal[invalid] = np.nan

    # If input was a single vector, return shape (3,)
    if Pmso.shape[0] == 1:
        return normal[0]
    elif Pmso.shape[0] == 1 and len(Pmso.shape) == 1:
        return normal[0]
    else:
        return normal


if __name__ == "__main__":
    Pmso = np.array(((0.0, 2.7, 0.0),(0.0, 2.0, 2.0)))
    normal = bs_normal(Pmso)

import numpy as np

def In_MPB(Pmso):
    """
    Determine if points are inside the Martian Magnetopause Boundary (MPB).

    Parameters
    ----------
    Pmso : ndarray of shape (N, 3)
        Position vectors in Mars–Solar–Orbital (MSO) coordinates [km].

    Returns
    -------
    flag : ndarray of shape (N,)
        Binary array: 1 if the point is inside the MPB, 0 otherwise.
    dr : ndarray of shape (N,)
        Radial distance difference (r - R_mp) in Mars radii (Rm),
        where negative values indicate inside the boundary.

    Notes
    -----
    The MPB shape is defined by an empirical model with two elliptical branches:
    one for the dayside (x >= 0) and one for the nightside (x < 0).

    Author: Chi Zhang (Python translation)
    """

    # Mars radius [km]
    Rm = 3390.0

    # Normalize positions by Mars radius
    x = Pmso[:, 0] / Rm
    r = np.sqrt(Pmso[:, 1]**2 + Pmso[:, 2]**2) / Rm

    # Initialize output
    flag = np.zeros_like(x, dtype=int)
    R_mb = np.zeros_like(x)

    # MPB empirical model parameters
    xF1, xF2 = 0.64, 1.60
    L1, L2 = 1.08, 0.528
    e1, e2 = 0.77, 1.009

    # Dayside: x >= 0
    idx_pos = x >= 0
    dx1 = x[idx_pos] - xF1
    R_mb[idx_pos] = (e1**2 - 1) * dx1**2 - 2 * e1 * L1 * dx1 + L1**2

    # Nightside: x < 0
    idx_neg = x < 0
    dx2 = x[idx_neg] - xF2
    R_mb[idx_neg] = (e2**2 - 1) * dx2**2 - 2 * e2 * L2 * dx2 + L2**2

    # Radial distance margin and flag
    R_bound = np.sqrt(R_mb)
    dr = r - R_bound
    flag[(r + 0.1) <= R_bound] = 1  # inside = 1

    return flag, dr


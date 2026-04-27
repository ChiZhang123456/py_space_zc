import numpy as np
from py_space_zc import ang
from py_space_zc.maven import bs_mpb_theta, bs_normal

def bs_normal_theta(Pmso):
    """
    Compute the bow shock normal vector based on the local SZA (solar zenith angle).

    Parameters
    ----------
    Pmso : ndarray
        Position(s) in MSO coordinates, shape (3,), (1,3), or (n,3), in units of Mars radius (Rm).

    Returns
    -------
    normal : ndarray
        Normal vector(s) on the bow shock surface corresponding to the given position(s).
        Same shape as Pmso. NaN if input is invalid or not on the model surface.
    """
    # Ensure Pmso is a 2D array
    Pmso = np.atleast_2d(Pmso)
    if Pmso.shape[1] != 3:
        raise ValueError("Input must have shape (3,), (1,3), or (n,3)")

    # Construct X direction for computing SZA
    xmso = np.zeros_like(Pmso)
    xmso[:, 0] = 1.0

    # Compute solar zenith angle
    sza = ang(Pmso, xmso)

    # Get model prediction from theta-based bow shock model
    res = bs_mpb_theta(sza)  # Should return dict with "xbs" and "Ryz_bs", shape (n,)

    # Project onto YZ plane and normalize direction
    yz_norm = np.sqrt(Pmso[:, 1]**2 + Pmso[:, 2]**2)
    # Avoid division by zero
    yz_norm[yz_norm == 0] = np.nan

    y_bs = res["Ryz_bs"] * (Pmso[:, 1] / yz_norm)
    z_bs = res["Ryz_bs"] * (Pmso[:, 2] / yz_norm)
    x_bs = res["xbs"]

    # Assemble the modeled bow shock surface points
    bs_points = np.column_stack((x_bs, y_bs, z_bs))

    # Compute normals at those surface points
    normal = bs_normal(bs_points)

    # If input was a single point, return shape (3,)

    return normal


if __name__ == "__main__":
    Pmso = np.array((0.785,  1.081, -1.692))
    print(bs_normal_theta(Pmso))
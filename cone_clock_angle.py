import numpy as np

def cone_clock_angle(B: np.ndarray):
    """
    Compute the clock angle and cone angle from magnetic field vectors.

    Parameters
    ----------
    B : np.ndarray
        Magnetic field vector(s), shape can be either (3,) for a single vector
        or (n, 3) for multiple vectors.

    Returns
    -------
    clock_angle : float or np.ndarray
        Clock angle(s) in degrees, defined as atan2(By, Bz), ranging from 0 to 360°.

    cone_angle : float or np.ndarray
        Cone angle(s) in degrees, defined as arccos(Bx / |B|), representing the angle
        between the magnetic field vector and the x-axis.
    """
    # Ensure B has shape (n, 3)
    B = np.atleast_2d(B)

    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    B_magnitude = np.linalg.norm(B, axis=1)


    cone_angle = np.rad2deg(np.arccos(np.clip(Bx / B_magnitude, -1.0, 1.0)))
    clock_angle = np.rad2deg(np.arctan2(By, Bz))
    clock_angle = np.where(clock_angle < 0, clock_angle + 360, clock_angle)

    # If input was a single vector, return scalars instead of arrays
    if B.shape[0] == 1:
        return clock_angle[0], cone_angle[0]
    else:
        return clock_angle, cone_angle


if __name__ == '__main__':
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(cone_clock_angle(B))
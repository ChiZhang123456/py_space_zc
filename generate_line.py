import numpy as np

def generate_line(x0, z0, angle_deg, length, npoints):
    """
    Generate a straight line centered at a given point and oriented at a specific angle.

    This function computes the (x, z) coordinates of `npoints` equally spaced along a straight line
    of total length `length`, centered at the point `(x0, z0)` and rotated by `angle_deg` degrees
    counterclockwise from the positive X-axis.

    It is commonly used to define sampling or slicing lines (e.g., in field or flow plots) where
    line integration, interpolation, or visualization is needed.

    Parameters
    ----------
    x0 : float
        The x-coordinate of the center point through which the line passes.
    z0 : float
        The z-coordinate of the center point through which the line passes.
    angle_deg : float
        Angle in degrees between the line and the +X axis.
        - 0 degrees means the line is horizontal.
        - 90 degrees means the line is vertical.
        - Positive angles rotate the line counterclockwise.
    length : float
        Total length of the line (from one end to the other).
    npoints : int
        Number of evenly spaced sample points along the line.

    Returns
    -------
    x_vals : ndarray of shape (npoints,)
        The x-coordinates of the generated line points.
    z_vals : ndarray of shape (npoints,)
        The z-coordinates of the generated line points.

    Notes
    -----
    The line is generated using a parametric form:
        x(t) = x0 + t * dx
        z(t) = z0 + t * dz
    where t ∈ [-0.5, 0.5] spans the line symmetrically about the center point.

    Example
    -------
    >>> x, z = generate_line(0, 0, 45, 10, 100)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, z)
    >>> plt.gca().set_aspect('equal')
    >>> plt.show()
    """
    # Convert angle from degrees to radians
    theta = np.deg2rad(angle_deg)

    # Compute the full extent of the line in x and z directions
    dx = np.cos(theta) * length
    dz = np.sin(theta) * length

    # Generate normalized positions along the line (from -0.5 to 0.5)
    t = np.linspace(-0.5, 0.5, npoints)

    # Compute final coordinates using the parametric form
    x_vals = x0 + t * dx
    z_vals = z0 + t * dz

    return x_vals, z_vals


if __name__ == '__main__':
    x0, z0 = 1, 2
    angle_deg = 45
    npoints = 1000
    length = 10
    x, z = generate_line(x0, z0, angle_deg, length, npoints)

    import matplotlib.pyplot as plt
    plt.plot(x, z)
    plt.show()
import numpy as np

def generate_plane(center, normal, width, height, nx=20, ny=20):
    """
    Generate a 3D plane mesh centered at a point with a given normal vector.

    Parameters
    ----------
    center : array-like of shape (3,)
        The (x0, y0, z0) coordinates of the center point of the plane.
    normal : array-like of shape (3,)
        The (nx, ny, nz) components of the normal vector of the plane.
    width : float
        Width of the plane in one in-plane direction (u-axis).
    height : float
        Height of the plane in the orthogonal in-plane direction (v-axis).
    nx : int, optional
        Number of grid points along the width.
    ny : int, optional
        Number of grid points along the height.

    Returns
    -------
    X, Y, Z : ndarray of shape (ny, nx)
        The 3D coordinates of the sampled plane mesh.
    """
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)  # normalize normal vector

    # Find two orthogonal vectors u, v in the plane
    if np.allclose(normal, [0, 0, 1]):
        u = np.cross(normal, [0, 1, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Generate grid in (s, t) plane
    s = np.linspace(-width/2, width/2, nx)
    t = np.linspace(-height/2, height/2, ny)
    S, T = np.meshgrid(s, t)

    # Construct the plane in 3D
    X = center[0] + S * u[0] + T * v[0]
    Y = center[1] + S * u[1] + T * v[1]
    Z = center[2] + S * u[2] + T * v[2]

    return X, Y, Z

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    center = [0, 0, 0]
    normal = [1, 1, 1]
    X, Y, Z = generate_plane_3d(center, normal, width=10, height=10, nx=30, ny=30)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7)
    ax.quiver(*center, *normal, length=5, color='r', label='Normal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


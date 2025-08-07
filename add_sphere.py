import matplotlib.pyplot as plt
import numpy as np

def add_sphere(ax=None, radius: float = 1.0):
    """
    Plot a filled 2D projection of a sphere (e.g., Mars) onto a matplotlib axis.

    This function draws a circle with shading to simulate a 3D spherical appearance,
    using light and dark gray hemispheres. It can be used to represent a planet in
    spacecraft or planetary physics visualizations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        The matplotlib axis to draw on. If None, the current axis will be used.
    radius : float, optional
        The radius of the sphere to draw (default is 1.0).

    Returns
    -------
    None
    """
    # Use current axis if no axis is provided
    if ax is None:
        ax = plt.gca()

    # Define angle array from -90° to +90° (front hemisphere)
    theta_front = np.linspace(-np.pi / 2, np.pi / 2, 3600)
    x_front = radius * np.cos(theta_front)
    y_front = radius * np.sin(theta_front)

    # Fill the front hemisphere with light gray to indicate illumination
    ax.fill(x_front, y_front, facecolor='#D3D3D3', edgecolor='none')

    # Define angle array from +90° to +270° (back hemisphere)
    theta_back = np.linspace(np.pi / 2, 3 * np.pi / 2, 3600)
    x_back = radius * np.cos(theta_back)
    y_back = radius * np.sin(theta_back)

    # Fill the back hemisphere with darker gray to indicate shading
    ax.fill(x_back, y_back, facecolor='gray', edgecolor='none')


# Add the sphere (Mars)
if __name__ == '__main__':
    fig, ax = plt.subplots()
    add_sphere(ax, radius = 1.0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

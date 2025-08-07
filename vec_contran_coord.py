import numpy as np

def sph2cart_vec(P, inp):
    """
    Convert magnetic field components from spherical to Cartesian coordinates for multiple points.
    
    Parameters:
    - P (numpy.ndarray): An Nx3 array where each row represents Cartesian position vectors [Px, Py, Pz].
    - inp (numpy.ndarray): An Nx3 array where each row contains spherical vector components [Br, Btheta, Bphi].

    Returns:
    - numpy.ndarray: An Nx3 array of the converted Cartesian vector components [Bx, By, Bz].
    """
    r = np.sqrt(P[:, 0]**2 + P[:, 1]**2 + P[:, 2]**2)
    theta = np.arccos(P[:, 2] / r)
    phi = np.arctan2(P[:, 1], P[:, 0])

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    out = np.zeros_like(inp)
    out[:, 0] = inp[:, 0] * sin_theta * cos_phi + inp[:, 1] * cos_theta * cos_phi - inp[:, 2] * sin_phi
    out[:, 1] = inp[:, 0] * sin_theta * sin_phi + inp[:, 1] * cos_theta * sin_phi + inp[:, 2] * cos_phi
    out[:, 2] = inp[:, 0] * cos_theta - inp[:, 1] * sin_theta

    return out


def cart2sph_vec(P, inp):
    """
    Convert vector components from Cartesian to spherical coordinates for multiple points.
    
    Parameters:
    - P (numpy.ndarray): An Nx3 array where each row represents Cartesian position vectors [Px, Py, Pz].
    - inp (numpy.ndarray): An Nx3 array where each row contains Cartesian vector components [Bx, By, Bz].

    Returns:
    - numpy.ndarray: An Nx3 array of the converted spherical vector components [Br, Btheta, Bphi].
    """
    r = np.sqrt(P[:, 0]**2 + P[:, 1]**2 + P[:, 2]**2)
    theta = np.arccos(P[:, 2] / r)
    phi = np.arctan2(P[:, 1], P[:, 0])

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    out = np.zeros_like(inp)
    out[:, 0] = inp[:, 0] * sin_theta * cos_phi + inp[:, 1] * sin_theta * sin_phi + inp[:, 2] * cos_theta
    out[:, 1] = inp[:, 0]  * cos_theta * cos_phi + inp[:, 1] * cos_theta * sin_phi - inp[:, 2] * sin_theta
    out[:, 2] = -inp[:, 0] * sin_phi + inp[:, 1] * cos_phi

    return out

def cart2pol_vec(P, inp):
    """
    Convert vector components from Cartesian to cylindrical coordinates for multiple points.
    
    Parameters:
    - P (numpy.ndarray): An Nx3 array where each row represents Cartesian position vectors [Px, Py, Pz].
    - inp (numpy.ndarray): An Nx3 array where each row contains Cartesian vector components [Bx, By, Bz].

    Returns:
    - numpy.ndarray: An Nx3 array of the converted cylindrical vector components [Brho, Bphi, Bz].
    """
    rho = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    phi = np.arctan2(P[:, 1], P[:, 0])

    out = np.zeros_like(inp)
    out[:, 0] = inp[:, 0] * np.cos(phi) + inp[:, 1] * np.sin(phi)
    out[:, 1] = -inp[:, 0] * np.sin(phi) + inp[:, 1] * np.cos(phi)
    out[:, 2] = inp[:, 2]

    return out


def pol2cart_vec(P, inp):
    """
    Convert vector components from cylindrical to Cartesian coordinates based on Cartesian coordinates of the position vector for multiple points.

    Parameters:
    - P (numpy.ndarray): An Nx3 array where each row represents Cartesian position vectors [Px, Py, Pz].
    - inp (numpy.ndarray): An Nx3 array where each row contains cylindrical vector components [Brho, Bphi, Bz].

    Returns:
    - numpy.ndarray: An Nx3 array of the converted Cartesian vector components [Bx, By, Bz].
    """
    rho = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
    phi = np.arctan2(P[:, 1], P[:, 0])

    out = np.zeros_like(inp)
    out[:, 0] = inp[:, 0] * np.cos(phi) - inp[:, 1] * rho * np.sin(phi)
    out[:, 1] = inp[:, 0] * np.sin(phi) + inp[:, 1] * rho * np.cos(phi)
    out[:, 2] = inp[:, 2]

    return out


if __name__ == "__main__":    
    
    P_cartesian = np.array([[1, 1, 1],
                        [0, 1, 0],
                        [-1, -1, -1]])

# Define corresponding data vectors (could be velocity, force, etc.)
    data_cartesian = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    data_sph = cart2sph_vec(P_cartesian, data_cartesian)
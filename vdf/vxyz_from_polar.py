import numpy as np
from scipy import constants
from .get_particle_mass_charge import get_particle_mass_charge

def vxyz_from_polar(energymat, phimat, thetamat, species):
    """
    Convert particle velocity coordinates from spherical (energy, theta, phi)
    to Cartesian components (Vx, Vy, Vz).

    Automatically detects whether theta is defined as:
      (1) Elevation angle from the XY plane: [-90°, +90°], or
      (2) Zenith angle from the +Z axis: [0°, 180°]

    Definitions:
    ------------
    - Elevation mode (common in some instruments like SWIA):
        * theta = 0° → velocity in XY plane
        * theta = +90° → velocity along +Z
        * theta = -90° → velocity along -Z

    - Zenith mode (common in spherical coordinates):
        * theta = 0° → velocity along +Z
        * theta = 90° → velocity in XY plane
        * theta = 180° → velocity along -Z

    Parameters:
    -----------
    energymat : np.ndarray
        Energy in eV, shape (nt, ne, nphi, ntheta)
    phimat : np.ndarray
        Azimuth angle in degrees, [0°, 360°)
    thetamat : np.ndarray
        Either elevation ([-90°, 90°]) or zenith ([0°, 180°]) angle, in degrees
    species : str
        Particle species (e.g., 'H+', 'O+', etc.)

    Returns:
    --------
    Vx, Vy, Vz : np.ndarray
        Velocity components in km/s
    """

    # Get mass and charge of species
    pmass, q = get_particle_mass_charge(species.lower())

    # Convert energy to speed in m/s: v = sqrt(2 * q * E / m)
    vmat = np.sqrt(2 * q * energymat / pmass)

    # Convert angles to radians
    theta_rad = np.deg2rad(thetamat)
    phi_rad = np.deg2rad(phimat)

    # Auto-detect theta style
    theta_min = np.nanmin(thetamat)
    theta_max = np.nanmax(thetamat)

    if theta_min < 0:
        # Elevation angle assumed
        # θ = 0 → XY plane, θ = +90 → +Z, θ = -90 → -Z
        Vx = vmat * np.cos(theta_rad) * np.cos(phi_rad) * 1e-3
        Vy = vmat * np.cos(theta_rad) * np.sin(phi_rad) * 1e-3
        Vz = vmat * np.sin(theta_rad) * 1e-3
    else:
        # Zenith angle assumed
        # θ = 0 → +Z, θ = 90 → XY plane, θ = 180 → -Z
        Vx = vmat * np.sin(theta_rad) * np.cos(phi_rad) * 1e-3
        Vy = vmat * np.sin(theta_rad) * np.sin(phi_rad) * 1e-3
        Vz = vmat * np.cos(theta_rad) * 1e-3

    return Vx, Vy, Vz

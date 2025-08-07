import numpy as np
from .get_particle_mass_charge import get_particle_mass_charge

def _1d(n, T, U, v, species):
    """
    Calculate the phase space density using a 1D Maxwellian distribution.

    Parameters
    ----------
    n : float
        Particle density in cm^-3
    T : float
        Temperature in eV
    U : float
        Bulk velocity in km/s
    v : float or np.ndarray
        Particle speed in km/s
    species : str
        Particle species ('H', 'e', 'O', 'O2')

    Returns
    -------
    f : float or np.ndarray
        Phase space density (PSD) in s/m^4

    Example
    -------
    >>> f = _1d(10, 10, -400, -100, 'H')
    """

    m, qe = get_particle_mass_charge(species.lower())
    # Convert units
    n = n * 1e6           # cm^-3 to m^-3
    u = np.array(u) * 1e3 # km/s to m/s
    V = np.array(V) * 1e3 # km/s to m/s

    # Thermal speed
    vth = np.sqrt(2 * qe * T / m)  # in m/s

    # Maxwellian phase space density
    f = n / (np.sqrt(np.pi) * vth) * np.exp(-((v - U) ** 2) / vth ** 2)

    return f

import numpy as np

def _2d(n, T, ux, uy, Vx, Vy, species):
    """
    Calculate the phase space density using a 2D drifting Maxwellian distribution.

    Parameters
    ----------
    n : float
        Particle density in cm^-3
    T : float
        Temperature in eV
    ux, uy : float
        Bulk velocity in x and y directions (km/s)
    Vx, Vy : float or np.ndarray
        Velocity components to evaluate PSD at (km/s)
    species : str
        Particle species ('H', 'e', 'O', 'O2')

    Returns
    -------
    f : float or np.ndarray
        Phase space density (PSD) in s^2/m^5

    Example
    -------
    >>> f = _2d(10, 10, -400, 0, -100, 0, 'H')
    """

    # Physical constants
    m, qe = get_particle_mass_charge(species.lower())

    # Convert units
    n = n * 1e6        # cm^-3 to m^-3
    ux = ux * 1e3      # km/s to m/s
    uy = uy * 1e3
    Vx = np.asarray(Vx) * 1e3
    Vy = np.asarray(Vy) * 1e3

    # Thermal speed
    vth = np.sqrt(2 * e * T / m)

    # Maxwellian PSD
    f = n / (np.pi * vth**2) * np.exp(-((Vx - ux)**2 + (Vy - uy)**2) / vth**2)

    return f


def _3d(n, T, u, Vx, Vy, Vz, species):
    """
    Calculate the 3D drifting Maxwellian phase space density.

    Parameters
    ----------
    n : float
        Density of particles in cm^-3
    T : float
        Temperature in eV
    u : array-like of float
        Bulk velocity [ux, uy, uz] in km/s
    Vx, Vy, Vz : float or np.ndarray
        Particle velocity components in km/s
    species : str
        Particle species: 'H', 'e', 'O', or 'O2'

    Returns
    -------
    f : float or np.ndarray
        Phase Space Density (PSD) in s^3/m^6
    """

    m, qe = get_particle_mass_charge(species.lower())

    # Convert units
    n = n * 1e6           # cm^-3 to m^-3
    u = np.array(u) * 1e3 # km/s to m/s
    Vx = np.array(Vx) * 1e3
    Vy = np.array(Vy) * 1e3
    Vz = np.array(Vz) * 1e3

    # Thermal speed
    vth = np.sqrt(2 * qe * T / m)  # in m/s

    # Relative velocity squared
    dV2 = (Vx - u[0])**2 + (Vy - u[1])**2 + (Vz - u[2])**2

    # Maxwellian PSD formula
    f = n * (np.pi * vth**2)**(-1.5) * np.exp(-dV2 / vth**2)

    return f


if __name__ == "__main__":
    # import numpy as np
    # import matplotlib.pyplot as plt

    n = 10  # cm^-3
    T = 10  # eV
    u = [-400, 0, 0]  # km/s
    Vx = 100
    Vy = 0
    Vz = 0
    # Vx = np.arange(-400, 400, 1)
    # Vy = np.zeros_like(Vx)
    # Vz = np.zeros_like(Vx)
    species = 'H'
    f = _3d(n, T, u, Vx, Vy, Vz, species)
    # plt.plot(Vx, f)
    # plt.yscale("log")
    # plt.show()
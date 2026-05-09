import numpy as np
from py_space_zc.vdf import get_particle_mass_charge


def _thermal_speed(T, species):
    """Return sqrt(2 kB T / m) in m/s for T in eV."""
    m, qe = get_particle_mass_charge(species.lower())
    return np.sqrt(2 * abs(qe) * T / m)


def _1d(n, T, U, V, species):
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
    V : float or np.ndarray
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

    # Convert units
    n = n * 1e6           # cm^-3 to m^-3
    u = np.array(U) * 1e3 # km/s to m/s
    v = np.array(V) * 1e3 # km/s to m/s

    # Thermal speed
    vth = _thermal_speed(T, species)

    # Maxwellian phase space density
    f = n / (np.sqrt(np.pi) * vth) * np.exp(-((v - u) ** 2) / vth ** 2)

    return f


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

    # Convert units
    n = n * 1e6        # cm^-3 to m^-3
    ux = ux * 1e3      # km/s to m/s
    uy = uy * 1e3
    Vx = np.asarray(Vx) * 1e3
    Vy = np.asarray(Vy) * 1e3

    # Thermal speed
    vth = _thermal_speed(T, species)

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

    # Convert units
    n = n * 1e6           # cm^-3 to m^-3
    u = np.array(u) * 1e3 # km/s to m/s
    Vx = np.array(Vx) * 1e3
    Vy = np.array(Vy) * 1e3
    Vz = np.array(Vz) * 1e3

    # Thermal speed
    vth = _thermal_speed(T, species)

    # Relative velocity squared
    dV2 = (Vx - u[0])**2 + (Vy - u[1])**2 + (Vz - u[2])**2

    # Maxwellian PSD formula
    f = n * (np.pi * vth**2)**(-1.5) * np.exp(-dV2 / vth**2)

    return f


def _bi_2d(n, T_perp, T_parallel, u_perp, u_parallel,
           V_perp, V_parallel, species):
    """
    Calculate a 2D gyrotropic drifting bi-Maxwellian in (V_perp, V_parallel).

    Parameters
    ----------
    n : float
        Particle density in cm^-3.
    T_perp, T_parallel : float
        Perpendicular and parallel temperatures in eV.
    u_perp, u_parallel : float
        Drift velocities in km/s.
    V_perp, V_parallel : float or np.ndarray
        Velocity coordinates in km/s.
    species : str
        Particle species ('H', 'e', 'O', 'O2').

    Returns
    -------
    f : float or np.ndarray
        Phase space density in s^3/m^6.
    """

    n = n * 1e6
    u_perp = np.asarray(u_perp) * 1e3
    u_parallel = np.asarray(u_parallel) * 1e3
    V_perp = np.asarray(V_perp) * 1e3
    V_parallel = np.asarray(V_parallel) * 1e3

    vth_perp = _thermal_speed(T_perp, species)
    vth_parallel = _thermal_speed(T_parallel, species)

    f = n / (np.pi ** 1.5 * vth_perp ** 2 * vth_parallel)
    f = f * np.exp(
        -((V_perp - u_perp) ** 2) / vth_perp ** 2
        -((V_parallel - u_parallel) ** 2) / vth_parallel ** 2
    )

    return f


def _bi_3d(n, T_perp, T_parallel, u, V_perp1, V_perp2,
           V_parallel, species):
    """
    Calculate a 3D drifting bi-Maxwellian with two perpendicular axes.

    Parameters
    ----------
    n : float
        Particle density in cm^-3.
    T_perp, T_parallel : float
        Perpendicular and parallel temperatures in eV.
    u : array-like of float
        Drift velocity [u_perp1, u_perp2, u_parallel] in km/s.
    V_perp1, V_perp2, V_parallel : float or np.ndarray
        Velocity coordinates in km/s.
    species : str
        Particle species ('H', 'e', 'O', 'O2').

    Returns
    -------
    f : float or np.ndarray
        Phase space density in s^3/m^6.
    """

    n = n * 1e6
    u = np.asarray(u) * 1e3
    V_perp1 = np.asarray(V_perp1) * 1e3
    V_perp2 = np.asarray(V_perp2) * 1e3
    V_parallel = np.asarray(V_parallel) * 1e3

    vth_perp = _thermal_speed(T_perp, species)
    vth_parallel = _thermal_speed(T_parallel, species)

    dV_perp2 = (V_perp1 - u[0]) ** 2 + (V_perp2 - u[1]) ** 2
    dV_parallel2 = (V_parallel - u[2]) ** 2

    f = n / (np.pi ** 1.5 * vth_perp ** 2 * vth_parallel)
    f = f * np.exp(-dV_perp2 / vth_perp ** 2
                   -dV_parallel2 / vth_parallel ** 2)

    return f


if __name__ == "__main__":
    # import numpy as np
    # import matplotlib.pyplot as plt

    n = 10  # cm^-3
    T = 52.198  # eV
    u = [-400, 0, 0]  # km/s
    Vx = -400
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

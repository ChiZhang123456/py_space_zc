import numpy as np
from scipy.special import gammaln

from py_space_zc.vdf import get_particle_mass_charge


def _thermal_speed(T, species):
    """Return sqrt(2 kB T / m) in m/s for T in eV."""
    m, qe = get_particle_mass_charge(species.lower())
    return np.sqrt(2 * abs(qe) * T / m)


def _gamma_ratio(a, b):
    """Return Gamma(a) / Gamma(b) with logarithmic scaling."""
    return np.exp(gammaln(a) - gammaln(b))


def _check_kappa(kappa):
    if np.any(np.asarray(kappa) <= 1.5):
        raise ValueError("kappa must be larger than 1.5 for this thermal-speed convention.")


def _1d(n, T, U, V, kappa, species):
    """
    Calculate a normalized 1D drifting kappa distribution.

    Parameters
    ----------
    n : float
        Particle density in cm^-3.
    T : float
        Temperature in eV.
    U : float
        Bulk velocity in km/s.
    V : float or np.ndarray
        Particle velocity in km/s.
    kappa : float
        Kappa index. Must be larger than 1.5.
    species : str
        Particle species ('H', 'e', 'O', 'O2').

    Returns
    -------
    f : float or np.ndarray
        Phase space density in s/m^4.
    """

    _check_kappa(kappa)

    n = n * 1e6
    u = np.asarray(U) * 1e3
    v = np.asarray(V) * 1e3
    vth = _thermal_speed(T, species)

    a2 = (kappa - 1.5) * vth ** 2
    prefactor = n * _gamma_ratio(kappa + 1, kappa + 0.5)
    prefactor = prefactor / (np.sqrt(np.pi * a2))

    f = prefactor * (1 + (v - u) ** 2 / a2) ** (-(kappa + 1))

    return f


def _3d(n, T, u, Vx, Vy, Vz, kappa, species):
    """
    Calculate an isotropic 3D drifting kappa distribution.

    This follows the common space-plasma convention
    v_kappa = sqrt((2*kappa - 3) / kappa * kB*T / m), with T in eV.

    Parameters
    ----------
    n : float
        Particle density in cm^-3.
    T : float
        Temperature in eV.
    u : array-like of float
        Bulk velocity [ux, uy, uz] in km/s.
    Vx, Vy, Vz : float or np.ndarray
        Velocity components in km/s.
    kappa : float
        Kappa index. Must be larger than 1.5.
    species : str
        Particle species ('H', 'e', 'O', 'O2').

    Returns
    -------
    f : float or np.ndarray
        Phase space density in s^3/m^6.
    """

    _check_kappa(kappa)

    n = n * 1e6
    u = np.asarray(u) * 1e3
    Vx = np.asarray(Vx) * 1e3
    Vy = np.asarray(Vy) * 1e3
    Vz = np.asarray(Vz) * 1e3
    vth = _thermal_speed(T, species)

    dV2 = (Vx - u[0]) ** 2 + (Vy - u[1]) ** 2 + (Vz - u[2]) ** 2
    a2 = (kappa - 1.5) * vth ** 2

    prefactor = n * _gamma_ratio(kappa + 1, kappa - 0.5)
    prefactor = prefactor / (np.pi ** 1.5 * a2 ** 1.5)

    f = prefactor * (1 + dV2 / a2) ** (-(kappa + 1))

    return f


def _bi(n, T_perp=None, T_parallel=None, vd=None, V_perp=None,
        V_parallel=None, kappa=None, species="e", **kwargs):
    """
    Calculate a drift bi-kappa distribution in (V_perp, V_parallel).

    Formula:
    K = n / [pi^(3/2) (kappa - 3/2)^(3/2) vth_parallel vth_perp^2]
        * Gamma(kappa + 1) / Gamma(kappa - 1/2)
        * [1 + 1/(kappa - 3/2)
             * (V_perp^2/vth_perp^2 + (V_parallel - vd)^2/vth_parallel^2)
          ]^(-(kappa + 1))

    Parameters
    ----------
    n : float
        Particle density in cm^-3.
    T_perp, T_parallel : float
        Perpendicular and parallel temperatures in eV.
    vd : float
        Field-aligned drift speed in km/s.
    V_perp, V_parallel : float or np.ndarray
        Perpendicular speed and parallel velocity in km/s.
    kappa : float
        Kappa index. Must be larger than 1.5.
    species : str, optional
        Particle species used to convert eV to thermal speed. Default is 'e'.

    Returns
    -------
    f : float or np.ndarray
        Full 3D gyrotropic phase space density in s^3/m^6.
    """

    vth_perp = kwargs.pop("vth_perp", None)
    vth_parallel = kwargs.pop("vth_parallel", None)
    if vth_perp is None:
        vth_perp = kwargs.pop("theta_perp", None)
    if vth_parallel is None:
        vth_parallel = kwargs.pop("theta_parallel", None)
    if kwargs:
        keys = ", ".join(kwargs)
        raise TypeError(f"Unexpected keyword argument(s): {keys}")
    if T_perp is not None and T_parallel is not None:
        vth_perp = _thermal_speed(T_perp, species) / 1e3
        vth_parallel = _thermal_speed(T_parallel, species) / 1e3
    if any(x is None for x in [vth_perp, vth_parallel, vd,
                               V_perp, V_parallel, kappa]):
        raise TypeError(
            "bi_kappa requires n, T_perp, T_parallel, vd, V_perp, "
            "V_parallel, kappa, and species. The legacy vth_perp and "
            "vth_parallel keywords are also accepted."
        )

    _check_kappa(kappa)

    n = n * 1e6
    vth_perp = np.asarray(vth_perp) * 1e3
    vth_parallel = np.asarray(vth_parallel) * 1e3
    vd = np.asarray(vd) * 1e3
    V_perp = np.asarray(V_perp) * 1e3
    V_parallel = np.asarray(V_parallel) * 1e3

    kappa_m32 = kappa - 1.5
    prefactor = n * _gamma_ratio(kappa + 1, kappa - 0.5)
    prefactor = prefactor / (
        np.pi ** 1.5 * kappa_m32 ** 1.5
        * vth_parallel * vth_perp ** 2
    )

    shape = 1 + (
        (V_perp ** 2 / vth_perp ** 2)
        + ((V_parallel - vd) ** 2 / vth_parallel ** 2)
    ) / kappa_m32

    f = prefactor * shape ** (-(kappa + 1))

    return f



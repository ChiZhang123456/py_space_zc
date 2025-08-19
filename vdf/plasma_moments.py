"""
Adapted from pyrfu's ts_skymap module:
(https://github.com/louis-richard/irfu-python), licensed under the MIT License.

Original code licensed under the MIT License.
Modified by Chi Zhang for compatibility with py_space_zc and to include
spacecraft velocity correction in the velocity moments calculation.
"""

import numpy as np
from .match_vdf_dims import match_vdf_dims
from .convert_energy_velocity import convert_energy_velocity
from .expand_4d_grid import expand_4d_grid
from .vxyz_from_polar import vxyz_from_polar
from .get_particle_mass_charge import get_particle_mass_charge
from scipy import constants


def plasma_moments(PSD, energymat, dEmat, phimat, thetamat,
                   dphimat, dthetamat, vsc, species):
    """
    Compute plasma moments (density, velocity, pressure, temperature,
    energy fluxes) from a 4D phase space density (PSD) array.

    Parameters
    ----------
    PSD : ndarray, shape (ntime, nenergy, nphi, ntheta)
        Phase space density [s^3/m^6].

    energymat : ndarray
        Particle energy [eV], corrected for spacecraft potential.
        Same shape as PSD.

    dEmat : ndarray
        Energy bin width [eV].
        Same shape as PSD.

    phimat : ndarray
        Azimuthal angle φ [deg].
        Same shape as PSD.

    thetamat : ndarray
        Polar angle θ [deg], defined as the angle between v and +Z axis.
        Same shape as PSD.

    dphimat : ndarray
        Azimuthal bin width [deg].
        Same shape as PSD.

    dthetamat : ndarray
        Polar bin width [deg].
        Same shape as PSD.

    vsc : ndarray, shape (ntime, 3)
        Spacecraft velocity vector in the instrument frame [m/s].
        Used to correct measured particle velocities.

    species : str
        Particle species, e.g., 'H+', 'O+', 'He+'.

    Returns
    -------
    dict
        Plasma moments:
        - 'n'   : number density [cm⁻³]
        - 'V'   : bulk velocity vector [km/s]
        - 'Pressure' : thermal pressure tensor [nPa]
        - 'P2'  : total pressure tensor (dynamic + thermal) [nPa]
        - 'Temp': temperature tensor [eV]
        - 'H'   : enthalpy flux vector [erg/s/cm²]
        - 'Q'   : heat flux vector [erg/s/cm²]
        - 'K'   : kinetic energy flux vector [erg/s/cm²]
    """

    num_time = PSD.shape[0]
    kb = constants.Boltzmann  # [J/K]

    # 1. Get particle mass [kg] and charge [C] from species
    pmass, qe = get_particle_mass_charge(species=species)

    # 2. Convert angles from degrees to radians
    thetamat = np.deg2rad(thetamat)
    phimat = np.deg2rad(phimat)
    dphimat = np.deg2rad(dphimat)
    dthetamat = np.deg2rad(dthetamat)

    # 3. Compute velocity magnitude from kinetic energy: E = 0.5 m v²
    Vt = np.sqrt(2 * energymat * qe / pmass)                  # [m/s]
    Vt_upper = np.sqrt(2 * (energymat + dEmat) * qe / pmass)  # [m/s]
    dvmat = Vt_upper - Vt                                     # Δv per bin [m/s]

    # 4. Convert spherical velocities to Cartesian components
    #    NOTE: theta = angle from +Z axis
    Vx = -Vt * np.sin(thetamat) * np.cos(phimat)
    Vy = -Vt * np.sin(thetamat) * np.sin(phimat)
    Vz = -Vt * np.cos(thetamat)

    # 4b. Spacecraft velocity correction
    #     Add spacecraft velocity to each particle velocity bin
    Vx += vsc[:, 0, None, None, None]
    Vy += vsc[:, 1, None, None, None]
    Vz += vsc[:, 2, None, None, None]

    # 5. Velocity space volume element: d³v = v² sinθ dv dΩ
    d3v = Vt**2 * np.sin(thetamat) * dvmat * dphimat * dthetamat  # [m³/s³]

    # 6. Zeroth moment: number density [1/m³]
    n_psd = np.nansum(PSD * d3v, axis=(1, 2, 3))

    # 7. First moment: bulk velocity vector [m/s]
    V_psd = np.zeros((num_time, 3))
    V_psd[:, 0] = np.nansum(Vx * PSD * d3v, axis=(1, 2, 3)) / n_psd
    V_psd[:, 1] = np.nansum(Vy * PSD * d3v, axis=(1, 2, 3)) / n_psd
    V_psd[:, 2] = np.nansum(Vz * PSD * d3v, axis=(1, 2, 3)) / n_psd

    # 8. Second moment: total momentum flux tensor P₂_ij [Pa]
    P2_psd = np.zeros((num_time, 3, 3))
    P2_psd[:, 0, 0] = np.nansum(pmass * Vx * Vx * PSD * d3v, axis=(1, 2, 3))
    P2_psd[:, 0, 1] = np.nansum(pmass * Vx * Vy * PSD * d3v, axis=(1, 2, 3))
    P2_psd[:, 0, 2] = np.nansum(pmass * Vx * Vz * PSD * d3v, axis=(1, 2, 3))
    P2_psd[:, 1, 1] = np.nansum(pmass * Vy * Vy * PSD * d3v, axis=(1, 2, 3))
    P2_psd[:, 1, 2] = np.nansum(pmass * Vy * Vz * PSD * d3v, axis=(1, 2, 3))
    P2_psd[:, 2, 2] = np.nansum(pmass * Vz * Vz * PSD * d3v, axis=(1, 2, 3))
    # Symmetrize tensor
    P2_psd[:, 1, 0] = P2_psd[:, 0, 1]
    P2_psd[:, 2, 0] = P2_psd[:, 0, 2]
    P2_psd[:, 2, 1] = P2_psd[:, 1, 2]

    # 9. Dynamic pressure tensor: n m V_i V_j
    dynP = pmass * n_psd[:, None, None] * V_psd[:, :, None] * V_psd[:, None, :]

    # 10. Thermal pressure tensor: P_th = P₂ - dynP
    P_psd = P2_psd - dynP

    # 11. Temperature tensor in eV
    T_psd = np.where(n_psd[:, None, None] != 0,
                     P_psd / (n_psd[:, None, None] * kb), np.nan)
    T_psd = T_psd * kb / qe  # [K] → [eV]

    # 12. Third moment: total energy flux vector [J/m²/s]
    M2_psd = np.zeros((num_time, 3))
    M2_psd[:, 0] = np.nansum(0.5 * pmass * Vx * Vt**2 * PSD * d3v, axis=(1, 2, 3))
    M2_psd[:, 1] = np.nansum(0.5 * pmass * Vy * Vt**2 * PSD * d3v, axis=(1, 2, 3))
    M2_psd[:, 2] = np.nansum(0.5 * pmass * Vz * Vt**2 * PSD * d3v, axis=(1, 2, 3))

    # 13. Kinetic energy flux: (1/2 n m V²) V
    Vabs2 = np.sum(V_psd**2, axis=1)
    K_psd = 0.5 * pmass * n_psd[:, None] * Vabs2[:, None] * V_psd

    # 14. Enthalpy flux: H_i = (trace(P)/2) V_i + Σ_j V_j P_ij
    Ptrace = P_psd[:, 0, 0] + P_psd[:, 1, 1] + P_psd[:, 2, 2]
    H_psd = np.zeros((num_time, 3))
    for i in range(3):
        H_psd[:, i] = (Ptrace / 2) * V_psd[:, i] + np.sum(V_psd * P_psd[:, i, :], axis=1)

    # 15. Heat flux: Q = M₂ - K - H
    Q_psd = M2_psd - K_psd - H_psd

    # 16. Unit conversions
    n_psd /= 1e6   # [1/m³] → [cm⁻³]
    V_psd /= 1e3   # [m/s] → [km/s]
    P_psd *= 1e9   # [Pa] → [nPa]
    P2_psd *= 1e9
    H_psd *= 1e3   # [J/m²/s] → [erg/s/cm²]
    Q_psd *= 1e3
    K_psd *= 1e3

    return {
        "n": n_psd,
        "V": V_psd,
        "Pressure": P_psd,
        "P2": P2_psd,
        "Temp": T_psd,
        "H": H_psd,
        "Q": Q_psd,
        "K": K_psd
    }

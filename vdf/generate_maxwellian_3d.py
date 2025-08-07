import numpy as np
from .convert_energy_velocity import convert_energy_velocity
from .maxwellian_distribution import _3d as maxwellian_3d
from .create_pdist_skymap import create_pdist_skymap
from pyrfu import pyrf

def generate_maxwellian_3d(n, U, T, time=None):
    """
    Generate a 3D Maxwellian distribution in ts_skymap format.

    Parameters
    ----------
    n : float
        Number density in cm^-3.
    U : array_like
        Bulk velocity vector [Ux, Uy, Uz] in km/s.
    T : float
        Temperature in eV.
    time : array_like of np.datetime64, optional
        Array of time values. If None, defaults to two fixed timestamps.

    Returns
    -------
    psd_vdf : pyrfu.ts_skymap
        Phase space density in ts_skymap format with units s^3/m^6.
    """
    if time is None:
        time = np.array([
            np.datetime64('2010-01-01T00:00:00'),
            np.datetime64('2010-01-01T01:00:00')
        ])

    # Define energy and angle grids
    energy = 10 ** np.linspace(np.log10(0.1), np.log10(30000), 50)  # eV
    phi = np.linspace(0, 360, 17)  # degrees (azimuth)
    theta = np.linspace(-90, 90, 9)  # degrees (elevation)

    # Construct velocity space
    energymat, phimat, thetamat = np.meshgrid(energy, phi, theta, indexing='ij')
    Vt = convert_energy_velocity(energymat, 'e>v', 'H')  # km/s
    Vx = Vt * np.cos(np.deg2rad(thetamat)) * np.cos(np.deg2rad(phimat))
    Vy = Vt * np.cos(np.deg2rad(thetamat)) * np.sin(np.deg2rad(phimat))
    Vz = Vt * np.sin(np.deg2rad(thetamat))

    # Compute Maxwellian distribution
    psd = maxwellian_3d(n, T, U, Vx, Vy, Vz, species="H")  # units: s^3/m^6

    # Duplicate for each time step
    psd = np.tile(psd[None, :, :, :], (len(time), 1, 1, 1))
    energy = np.tile(energy[None, :], (len(time), 1))

    # Create ts_skymap
    psd_vdf = create_pdist_skymap(
        time=time,
        energy=energy,
        data=psd,
        phi=phi,
        theta=theta,
        Units="s^3/m^6",
        species="H",
        direction_is_velocity=True
    )

    return psd_vdf

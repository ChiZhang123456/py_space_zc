import numpy as np
from .convert_energy_velocity import convert_energy_velocity
from .match_vdf_dims import match_vdf_dims

def expand_4d_grid(psd, energy, phi, theta, particle_type, **kwargs):
    """
    Expand and align velocity-space coordinates (energy, phi, theta) into
    4D grids for volume integration of the phase space density (PSD),
    and compute the differential velocity-space volume element d³v.

    Author:
    -------
    Chi Zhang

    Purpose:
    --------
    This function is designed for preparing velocity space grids that
    match the dimensions of a 4D phase space density array. It calculates
    the corresponding differential volume element `d³v` in velocity space
    required for integrating physical moments such as number density,
    bulk velocity, pressure, and heat flux from PSD.

    Usage:
    ------
    >>> from expand_4d_grid import expand_4d_grid
    >>> result = expand_4d_grid(psd, energy, phi, theta, particle_type='O+')
    >>> d3v = result['d3v']
    >>> # Integration: n = ∑ (PSD * d3v)

    Parameters:
    -----------
    psd : np.ndarray
        4D array of phase space density with shape:
        (num_time, num_energy, num_phi, num_theta)
        Units: [s^3/m^6]

    energy : np.ndarray
        Particle energy in eV. Shape can be:
        - (num_energy,) for static energy bins
        - (num_time, num_energy) for time-varying spectra

    phi : np.ndarray
        Azimuthal angles in degrees. Shape:
        - (num_phi,) or (num_time, num_phi)

    theta : np.ndarray
        Polar angles in degrees. Shape:
        - (num_theta,), (num_time, num_theta),
          or (num_time, num_energy, num_theta)

    particle_type : str
        String indicating particle species. Supported options:
        'H', 'H+', 'p', 'He++', 'O', 'O2', 'CO2', 'e'

    Optional Keyword Arguments:
    ---------------------------
    delta_theta or dtheta : np.ndarray
        Bin width of polar angles in degrees.
        Shape: (num_time, num_energy, num_theta).
        If not provided, bin widths are estimated from θ.

    Returns:
    --------
    dict with the following fields:

        energymat   : np.ndarray
            Expanded energy grid [eV], shape = (t, e, φ, θ)

        dEmat       : np.ndarray
            Energy bin width [eV], same shape

        phimat      : np.ndarray
            Azimuthal angle grid [deg], shape = (t, e, φ, θ)

        thetamat    : np.ndarray
            Polar angle grid [deg], shape = (t, e, φ, θ)

        dvmat       : np.ndarray
            Differential velocity [m/s] between bin edges, shape = (t, e, φ, θ)

        deltaphi    : np.ndarray
            Azimuthal bin width [deg], shape = (t, e, φ, θ)

        deltatheta  : np.ndarray
            Polar bin width [deg], shape = (t, e, φ, θ)

        d3v         : np.ndarray
            Differential velocity-space volume element [m³/s³], shape = (t, e, φ, θ)

    Notes:
    ------
    - Converts energy [eV] to velocity [m/s] using `convert_energy_velocity`.
    - Automatically expands all input variables to match PSD shape.
    - Uses ∆θ and ∆φ for solid-angle-weighted velocity integration.
    - Integration formula used:

        d³v = v² * cos(θ_rad) * dv * dθ_rad * dφ_rad

    - Units:
        PSD:     [s³/m⁶]
        d3v:     [m³/s³]
        PSD*d3v: dimensionless → number density [m⁻³] when integrated

    """

    # Match dimensional shapes with PSD shape
    energy_new, dE_new, phi_new, theta_new = match_vdf_dims(psd, energy, phi, theta)
    num_time, num_energy, num_phi, num_theta = psd.shape

    # Expand each quantity to 4D
    energymat = np.tile(energy_new[:, :, None, None], (1, 1, num_phi, num_theta))
    dEmat     = np.tile(dE_new[:, :, None, None],     (1, 1, num_phi, num_theta))
    phimat    = np.tile(phi_new[:, None, :, None],    (1, num_energy, 1, num_theta))
    thetamat  = np.tile(theta_new[:, :, None, :],     (1, 1, num_phi, 1))

    # Convert energy [eV] → velocity [m/s]
    v      = convert_energy_velocity(energymat, 'e2v', particle_type) * 1e3
    vupper = convert_energy_velocity(energymat + dEmat, 'e2v', particle_type) * 1e3
    dvmat  = vupper - v

    # Get delta_theta [deg] and expand to 4D
    if "delta_theta" in kwargs or "dtheta" in kwargs:
        theta_key = "delta_theta" if "delta_theta" in kwargs else "dtheta"
        deltatheta = kwargs[theta_key]  # shape: (t, e, θ)
        deltatheta = np.tile(deltatheta[:, :, None, :], (1, 1, num_phi, 1))
    else:
        # Estimate bin width from θ grid
        deltatheta = np.zeros_like(thetamat)
        deltatheta[:, :, :, :-1] = thetamat[:, :, :, 1:] - thetamat[:, :, :, :-1]
        deltatheta[:, :, :, -1]  = deltatheta[:, :, :, -2]  # replicate last bin

    # Compute delta_phi [deg]
    deltaphi = np.ones_like(phimat) * np.median(phimat[:, :, 1:, :] - phimat[:, :, :-1, :])

    # Ensure positive bin widths
    deltaphi    =  np.abs(deltaphi)      # [deg]
    deltatheta  =  np.abs(deltatheta)    # [deg]

    # Convert θ to radians for cos(θ)
    thetamat_rad = np.deg2rad(thetamat)

    # Compute d³v using solid angle in spherical coordinates
    d3v = v ** 2 * np.sin(thetamat_rad) * dvmat * np.deg2rad(deltatheta) * np.deg2rad(deltaphi)

    return {
        "energymat": energymat,       # [eV]
        "dEmat": dEmat,               # [eV]
        "phimat": phimat,             # [deg]
        "thetamat": thetamat,         # [deg]
        "dvmat": dvmat,               # [m/s]
        "deltaphi": deltaphi,         # [deg]
        "deltatheta": deltatheta,     # [deg]
        "d3v": d3v                    # [m³/s³]
    }

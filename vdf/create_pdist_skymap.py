from .match_vdf_dims import match_vdf_dims
from .expand_4d_grid import expand_4d_grid
from py_space_zc import ts_skymap
import numpy as np

def create_pdist_skymap(
    time,
    energy,
    data,
    phi,
    theta,
    Units="keV/(cm^2 s sr keV)",
    species="H+",
    direction_is_velocity=True,
    deltatheta=None,
):
    """
    Create a `pyrfu`-compatible time series skymap object from phase space density data.

    Parameters
    ----------
    time : np.datetime64 or list of datetime64
        Single timestamp or array of timestamps corresponding to the PSD measurements.

    energy : np.ndarray
        Energy array in eV. Can be 1D (n_energy,) or 2D (n_time, n_energy).

    data : np.ndarray
        4D phase space density or differential energy flux array.
        Shape must be (n_time, n_energy, n_phi, n_theta).

    phi : np.ndarray
        Azimuth angle array [deg]. Can be (n_phi,) or (n_time, n_phi).

    theta : np.ndarray
        Polar angle array [deg]. Can be:
        - (n_theta,), (n_time, n_theta), or (n_time, n_energy, n_theta)

    Units : str, optional
        Unit label to assign to the output (default is "keV/(cm² s sr keV)").

    species : str, optional
        Particle species label (default: "H+").

    direction_is_velocity : bool, optional
        If True, assumes input direction is particle velocity direction.
        Applies directional correction: flips theta and shifts phi by 180° to standardize conventions.

    deltatheta : np.ndarray, optional
        Bin widths for polar angles (same shape as theta). If provided, included in attrs["dtheta"].

    Returns
    -------
    vdf : pyrfu ts_skymap object
        A `pyrfu.pyrf.ts_skymap` object with the formatted PSD data, energy, and angle grids.
        This object can be directly plotted using `vdf.plot()` or processed further.
    """

    # Apply angular convention corrections if input is in velocity direction frame
    if direction_is_velocity:
        phi -= 180
        phi[phi <= 0] += 360     # Wrap into [0, 360]
        theta = -theta           # Flip theta if from velocity frame

    # Match energy, phi, theta to match PSD/data dimensions
    energy_new, dE_new, phi_new, theta_new = match_vdf_dims(data, energy, phi, theta)

    # Correct theta if it is in [-90, 90] (e.g., SWIA)
    if np.min(theta_new) < 0 and np.max(theta_new) <= 120:
        theta_new = 90 - theta_new   # Convert to [0, 180]

    # Prepare attribute dictionary
    attrs = {"UNITS": Units}
    if deltatheta is not None and isinstance(deltatheta, np.ndarray):
        attrs["dtheta"] = deltatheta

    # Create pyrfu ts_skymap object
    vdf = ts_skymap(
        time=time,
        data=data,
        energy=energy_new,
        phi=phi_new,
        theta=theta_new,
        attrs=attrs,
        glob_attrs={
            "species": species,
        },
    )

    return vdf

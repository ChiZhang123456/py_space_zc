from .match_vdf_dims import match_vdf_dims
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
    copy_data=True,   # <--- NEW: control whether to copy `data`
):
    """
    Create a `pyrfu`-compatible skymap object from PSD/DEF data.

    Notes on copying / aliasing
    ---------------------------
    - `phi`/`theta` are copied locally before modification -> caller not affected.
    - If `copy_data=True`, we copy `data` before passing to ts_skymap to avoid aliasing.
      Set to False only if you are OK with `vdf.data` sharing memory with the input.
    - `deltatheta` is copied into attrs to avoid aliasing between caller and `vdf.attrs`.
    """

    # Make local, writable float copies for angular edits (cheaper & clearer than deepcopy)
    new_phi   = np.array(phi,   copy=True, dtype=float)
    new_theta = np.array(theta, copy=True, dtype=float)

    if direction_is_velocity:
        new_phi -= 180.0
        new_phi[new_phi <= 0.0] += 360.0   # wrap into (0, 360]
        new_theta = -new_theta             # flip for velocity-frame convention

    # Match grids to data dims (returns new arrays; inputs are not modified)
    energy_new, dE_new, phi_new, theta_new = match_vdf_dims(data, energy, new_phi, new_theta)

    # Correct theta if in [-90, 90] range (e.g., SWIA-like convention)
    if np.min(theta_new) < 0 and np.max(theta_new) <= 120:
        theta_new = 90.0 - theta_new

    # Prepare attributes; copy dtheta to avoid aliasing with caller
    attrs = {"UNITS": Units}
    if deltatheta is not None:
        attrs["dtheta"] = np.array(deltatheta, copy=True)

    # Optionally copy `data` to avoid sharing memory with the caller
    data_to_use = np.array(data, copy=True) if copy_data else data

    vdf = ts_skymap(
        time=time,
        data=data_to_use,
        energy=energy_new,
        phi=phi_new,
        theta=theta_new,
        attrs=attrs,
        glob_attrs={"species": species},
    )

    return vdf

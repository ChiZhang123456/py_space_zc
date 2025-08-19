"""
Adapted from pyrfu's ts_skymap module:
(https://github.com/louis-richard/irfu-python), licensed under the MIT License

Original code licensed under the MIT License.
Modified by Chi Zhang for compatibility with py_space_zc.
"""
# 3rd party imports
import numpy as np
import xarray as xr

def ts_skymap(time, data, energy, phi, theta, **kwargs):
    r"""Creates a skymap of the distribution function with optional time-dependent phi/theta.

    Parameters
    ----------
    time : np.ndarray
        List of times. Shape: (n_time,)
    data : np.ndarray
        Distribution function. Shape: (n_time, n_energy, n_phi, n_theta)
    energy : np.ndarray
        Energy levels. Shape: (n_time, n_energy) or (n_energy,)
    phi : np.ndarray
        Azimuth angles. Shape: (n_time, n_phi) or (n_phi,)
    theta : np.ndarray
        Elevation angles. Shape can be:
            - (n_theta,)
            - (n_time, n_theta)
            - (n_time, n_energy, n_theta)

       Other Parameters
       ----------------
       **kwargs
           Keyword arguments:
               - energy0 : array_like
               - energy1 : array_like
               - esteptable : array_like
               - attrs : dict
               - coords_attrs : dict
               - glob_attrs : dict

    Returns
    -------
    out : xarray.Dataset
        Skymap of the distribution function.

    """

    # Check input type
    assert isinstance(time, np.ndarray), "time must be numpy.ndarray"
    assert isinstance(data, np.ndarray), "data must be numpy.ndarray"
    assert isinstance(energy, np.ndarray), "energy must be numpy.ndarray"
    assert isinstance(phi, np.ndarray), "phi must be numpy.ndarray"
    assert isinstance(theta, np.ndarray), "theta must be numpy.ndarray"

    # Index coordinates
    n_time, n_energy, n_phi, n_theta = data.shape

    # Handle optional parameters
    energy0 = kwargs.get("energy0", energy[0, :] if energy.ndim == 2 else energy)
    energy1 = kwargs.get("energy1", energy[1, :] if energy.ndim == 2 else energy)
    esteptable = kwargs.get("esteptable", np.zeros(n_time, dtype=np.uint8))
    attrs = kwargs.get("attrs", {})
    glob_attrs = kwargs.get("glob_attrs", {})
    coords_attrs = kwargs.get("coords_attrs", {})

    assert isinstance(esteptable, np.ndarray) and esteptable.ndim == 1
    assert isinstance(attrs, dict)
    assert isinstance(glob_attrs, dict)
    assert isinstance(coords_attrs, dict)

    # Check esteptable
    assert isinstance(esteptable, np.ndarray), "esteptable must be 1D numpy.ndarray"
    assert esteptable.ndim == 1, "esteptable must be 1D numpy.ndarray"
    assert esteptable.shape[0] == len(time), "esteptable is not consistent with time"

    attrs = kwargs.get("attrs", {})
    coords_attrs = kwargs.get("coords_attrs", {})
    glob_attrs = kwargs.get("glob_attrs", {})

    # Check attributes are dictionaries
    assert isinstance(attrs, dict)

    out_dict = {
        "data": (["time", "idx0", "idx1", "idx2"], data),
        "time": ("time", time),
        "idx0": ("idx0", np.arange(n_energy)),
        "idx1": ("idx1", np.arange(n_phi)),
        "idx2": ("idx2", np.arange(n_theta)),
    }

    # Energy: either [time, energy] or [energy]
    if energy.ndim == 2:
        out_dict["energy"] = (["time", "idx0"], energy)
    else:
        out_dict["energy"] = (["idx0"], energy)

    # Phi: [time, phi] or [phi]
    if phi.ndim == 2:
        out_dict["phi"] = (["time", "idx1"], phi)
    else:
        out_dict["phi"] = (["idx1"], phi)

    # Theta coordinate (auto-handle 1D / 2D / 3D)
    if theta.ndim == 3:
        out_dict["theta"] = (["time", "idx0", "idx2"], theta)
    elif theta.ndim == 2:
        out_dict["theta"] = (["time", "idx2"], theta)
    elif theta.ndim == 1:
        out_dict["theta"] = (["idx2"], theta)
    else:
        raise ValueError("theta must have 1D, 2D, or 3D shape")

    # Construct global attributes and sort them
    # remove energy0, energy1, and esteptable from global attrs to overwrite
    overwrite_keys = ["energy0", "energy1", "esteptable"]
    glob_attrs = {k: glob_attrs[k] for k in glob_attrs if k not in overwrite_keys}
    glob_attrs = {
        "energy0": energy0,
        "energy1": energy1,
        "esteptable": esteptable,
        **glob_attrs,
    }

    glob_attrs = {k: glob_attrs[k] for k in sorted(glob_attrs)}

    # Create Dataset
    out = xr.Dataset(out_dict, attrs=glob_attrs)

    # Sort and fill coordinates attributes
    for k in coords_attrs:
        out[k].attrs = {m: coords_attrs[k][m] for m in sorted(coords_attrs[k])}

    # Sort and fill data attributes
    out.data.attrs = {k: attrs[k] for k in sorted(attrs)}

    return out

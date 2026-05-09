import numpy as np

from .fit_omni_1d import energy_to_speed
from .flux_convert import flux_convert


def _theta_integral(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2,):
        raise ValueError("theta must contain two limits in radians.")

    theta1, theta2 = theta
    theta_min = min(theta1, theta2)
    theta_max = max(theta1, theta2)

    if theta_min < 0.0:
        return abs(np.sin(theta2) - np.sin(theta1)), "cos"
    if theta_min >= 0.0 and theta_max <= np.pi:
        return abs(np.cos(theta1) - np.cos(theta2)), "sin"

    return abs(np.sin(theta2) - np.sin(theta1)), "cos"


def _as_dataarray(energy, DEF, species):
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required when DEF is not already a DataArray.") from exc

    energy = np.asarray(energy, dtype=float)
    DEF = np.asarray(DEF, dtype=float)

    if energy.ndim != 1:
        raise ValueError("energy must be 1D when DEF is not an xarray.DataArray.")
    input_was_1d = DEF.ndim == 1
    if input_was_1d:
        DEF = np.stack([DEF, DEF], axis=0)

    if DEF.shape[-1] != energy.size:
        raise ValueError("The last dimension of DEF must have the same length as energy.")

    dims = [f"dim_{idx}" for idx in range(DEF.ndim)]
    dims[-1] = "energy"
    coords = {"energy": energy}
    out = xr.DataArray(DEF, dims=dims, coords=coords)
    out.attrs["species"] = species
    out.attrs["UNITS"] = "keV/(cm^2 s sr keV)"
    return out, input_was_1d


def _is_dataarray_like(inp):
    return hasattr(inp, "attrs") and hasattr(inp, "dims") and hasattr(inp, "energy")


def _parse_inputs(energy_or_omni, DEF, species, phi, theta):
    if DEF is not None and species is not None and isinstance(species, str):
        return energy_or_omni, DEF, species, phi, theta, False

    if _is_dataarray_like(energy_or_omni):
        omni = energy_or_omni
        energy = omni.energy
        def_data = omni
        species_attr = omni.attrs.get("species")

        if DEF is not None:
            phi = DEF
        if species is not None:
            if isinstance(species, str):
                species_attr = species
            else:
                theta = species
        if species_attr is None:
            raise ValueError("species is missing. Set omni.attrs['species'] or pass it as a string.")

        return energy, def_data, species_attr, phi, theta, True

    if DEF is None or species is None:
        raise TypeError(
            "Use omni_def2psd(omni, phi, theta) or "
            "omni_def2psd(energy, DEF, species, phi=..., theta=...)."
        )
    return energy_or_omni, DEF, species, phi, theta, False


def omni_def2psd(energy, DEF=None, species=None, phi=2 * np.pi,
                 theta=(-np.pi / 4, np.pi / 4)):
    """
    Convert DEF to omni reduced PSD for 1D fitting.

    Parameters
    ----------
    energy : array-like or xarray.DataArray
        Energy grid in eV, or a ts_spectr/xarray.DataArray containing DEF,
        energy coordinate, UNITS, and species metadata.
    DEF : array-like or xarray.DataArray
        Differential energy flux. If the first argument is a DataArray, this
        positional argument is interpreted as phi.
    species : str
        Particle species, for example "H+", "O+", or "e". If the first
        argument is a DataArray, a non-string third positional argument is
        interpreted as theta.
    phi : float, default 2*pi
        Azimuthal integration range in radians.
    theta : sequence of two floats, default (-pi/4, pi/4)
        Polar or elevation angle limits in radians. Negative limits are
        interpreted as elevation angle from the XY plane and use cos(theta).
        Non-negative limits within [0, pi] are interpreted as polar angle from
        +Z and use sin(theta).

    Returns
    -------
    psd_omni : ndarray or xarray.DataArray
        Omni reduced PSD, f(v) v^2 dOmega, with units s/m^4.
    """

    energy, DEF, species, phi, theta, input_is_spectr = _parse_inputs(
        energy, DEF, species, phi, theta
    )

    if hasattr(DEF, "attrs") and hasattr(DEF, "dims"):
        def_da = DEF.copy()
        def_da.attrs.setdefault("species", species)
        def_da.attrs.setdefault("UNITS", "keV/(cm^2 s sr keV)")
        input_was_1d = def_da.ndim == 1
        if input_was_1d:
            import xarray as xr
            def_da = xr.concat([def_da, def_da], dim="dim_0")
    else:
        def_da, input_was_1d = _as_dataarray(energy, DEF, species)

    psd = flux_convert(def_da, option="def2psd")
    speed = energy_to_speed(energy, species) * 1e3
    theta_factor, theta_weight = _theta_integral(theta)
    solid_angle_factor = float(phi) * theta_factor
    psd_omni = psd * speed**2 * solid_angle_factor
    if input_was_1d:
        psd_omni = psd_omni.isel(dim_0=0)

    if hasattr(psd_omni, "attrs"):
        psd_omni.attrs["UNITS"] = "s/m^4"
        psd_omni.attrs["theta_weight"] = theta_weight
        psd_omni.attrs["theta_limits_rad"] = tuple(np.asarray(theta, dtype=float))
        psd_omni.attrs["phi_rad"] = float(phi)

    if input_is_spectr or (hasattr(DEF, "attrs") and hasattr(DEF, "dims")):
        return psd_omni
    return np.asarray(psd_omni.data)

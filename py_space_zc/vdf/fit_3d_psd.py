import numpy as np
from scipy.optimize import least_squares

from .get_particle_mass_charge import get_particle_mass_charge
from .match_vdf_dims import match_vdf_dims
from .maxwellian_distribution import _3d as maxwellian_3d
from .kappa_distribution import _3d as kappa_3d


def _energy_angles_to_vxyz(energy, phi, theta, species):
    mass, charge = get_particle_mass_charge(species)
    speed = np.sqrt(2.0 * abs(charge) * np.asarray(energy, dtype=float) / mass)
    speed = speed * 1e-3

    phi_rad = np.deg2rad(phi)
    theta_rad = np.deg2rad(theta)
    if np.nanmin(theta) < 0:
        vx = speed * np.cos(theta_rad) * np.cos(phi_rad)
        vy = speed * np.cos(theta_rad) * np.sin(phi_rad)
        vz = speed * np.sin(theta_rad)
    else:
        vx = speed * np.sin(theta_rad) * np.cos(phi_rad)
        vy = speed * np.sin(theta_rad) * np.sin(phi_rad)
        vz = speed * np.cos(theta_rad)
    return vx, vy, vz


def _expand_energy_angles(psd_data, energy, phi, theta):
    energy_new, _, phi_new, theta_new = match_vdf_dims(psd_data, energy, phi, theta)
    nt, ne, nphi, ntheta = psd_data.shape
    energymat = np.tile(energy_new[:, :, None, None], (1, 1, nphi, ntheta))
    phimat = np.tile(phi_new[:, None, :, None], (1, ne, 1, ntheta))
    thetamat = np.tile(theta_new[:, :, None, :], (1, 1, nphi, 1))
    return energymat, phimat, thetamat


def _get_attr(obj, name, default=None):
    if hasattr(obj, "attrs") and name in obj.attrs:
        return obj.attrs[name]
    return default


def _extract_from_skymap(psd):
    psd_data = np.asarray(psd.data, dtype=float)
    if psd_data.ndim != 4:
        raise ValueError("skymap PSD data must be 4D: time, energy, phi, theta.")

    energy = np.asarray(psd.energy.data, dtype=float)
    phi = np.asarray(psd.phi.data, dtype=float)
    theta = np.asarray(psd.theta.data, dtype=float)
    species = _get_attr(psd, "species", None)
    if species is None:
        species = _get_attr(psd, "Species", None)
    if species is None:
        species = "H+"
    time = np.asarray(psd.time.data) if hasattr(psd, "time") else None

    energymat, phimat, thetamat = _expand_energy_angles(
        psd_data, energy, phi, theta
    )
    vx, vy, vz = _energy_angles_to_vxyz(energymat, phimat, thetamat, species)
    return psd_data, vx, vy, vz, species, time


def _prepare_array_inputs(psd, vx, vy, vz, species):
    psd_data = np.asarray(psd, dtype=float)
    if psd_data.ndim == 3:
        psd_data = psd_data[None, ...]
    if psd_data.ndim != 4:
        raise ValueError("psd must be 3D or 4D when vx, vy, vz are provided.")

    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)
    if vx.ndim == 3:
        vx = vx[None, ...]
    if vy.ndim == 3:
        vy = vy[None, ...]
    if vz.ndim == 3:
        vz = vz[None, ...]
    if vx.shape != psd_data.shape or vy.shape != psd_data.shape or vz.shape != psd_data.shape:
        raise ValueError("vx, vy, vz must have the same shape as psd.")
    return psd_data, vx, vy, vz, species, None


def maxwellian_3d_psd_model(vx, vy, vz, n, T, ux=0.0, uy=0.0, uz=0.0,
                            species="H+"):
    """
    Evaluate an isotropic drifting 3D Maxwellian PSD.

    vx, vy, and vz are in km/s. The returned PSD has units s^3/m^6.
    Density n is in cm^-3 and scalar temperature T is in eV.
    """

    return maxwellian_3d(n, T, [ux, uy, uz], vx, vy, vz, species)


def kappa_3d_psd_model(vx, vy, vz, n, T, ux=0.0, uy=0.0, uz=0.0,
                       kappa=5.0, species="H+"):
    """
    Evaluate an isotropic drifting 3D kappa PSD.

    vx, vy, and vz are in km/s. The returned PSD has units s^3/m^6.
    Density n is in cm^-3, scalar temperature T is in eV, and kappa > 1.5.
    """

    return kappa_3d(n, T, [ux, uy, uz], vx, vy, vz, kappa, species)


def _normalize_mode(mode):
    if mode is None:
        mode = ["maxwellian"]
    if isinstance(mode, str):
        mode = [mode]
    aliases = {
        "maxwellian": "maxwellian",
        "m": "maxwellian",
        "kappa": "kappa",
        "k": "kappa",
    }
    normalized = []
    for item in mode:
        key = item.lower().strip()
        if key not in aliases:
            raise ValueError("mode entries must be 'maxwellian' or 'kappa'.")
        normalized.append(aliases[key])
    if len(normalized) == 0:
        raise ValueError("mode must contain at least one component.")
    return normalized


def _component_param_count(component):
    if component == "maxwellian":
        return 5
    if component == "kappa":
        return 6
    raise ValueError(f"Unknown component: {component}")


def _evaluate_mixture(vx, vy, vz, params, species, mode):
    total = np.zeros_like(vx, dtype=float)
    idx = 0
    for component in mode:
        if component == "maxwellian":
            n, T, ux, uy, uz = params[idx:idx + 5]
            total += maxwellian_3d_psd_model(vx, vy, vz, n, T, ux, uy, uz,
                                             species)
            idx += 5
        elif component == "kappa":
            n, T, ux, uy, uz, kappa = params[idx:idx + 6]
            total += kappa_3d_psd_model(vx, vy, vz, n, T, ux, uy, uz, kappa,
                                        species)
            idx += 6
    return total


def _estimate_initial_single(vx, vy, vz, psd, good, species):
    weights = psd[good].astype(float)
    weights = weights / np.sum(weights)
    x = vx[good]
    y = vy[good]
    z = vz[good]
    ux = float(np.sum(weights * x))
    uy = float(np.sum(weights * y))
    uz = float(np.sum(weights * z))
    dv2 = (x - ux) ** 2 + (y - uy) ** 2 + (z - uz) ** 2
    mass, charge = get_particle_mass_charge(species)
    T = mass * np.sum(weights * dv2) * 1e6 / (3.0 * 2.0 * abs(charge))
    T = float(max(T, 1e-3))

    peak_idx = np.nanargmax(np.where(good, psd, np.nan))
    unit = maxwellian_3d_psd_model(
        vx.ravel()[peak_idx],
        vy.ravel()[peak_idx],
        vz.ravel()[peak_idx],
        1.0,
        T,
        ux,
        uy,
        uz,
        species,
    )
    n = float(psd.ravel()[peak_idx] / unit) if unit > 0 else 1.0
    return max(n, 1e-12), T, ux, uy, uz


def _initial_guess(vx, vy, vz, psd, good, species, mode):
    n0, T0, ux0, uy0, uz0 = _estimate_initial_single(vx, vy, vz, psd, good,
                                                     species)
    p0 = []
    for idx, component in enumerate(mode):
        scale = 1.0 + 0.8 * idx
        n_comp = n0 / len(mode)
        if component == "maxwellian":
            p0.extend([n_comp, T0 * scale, ux0, uy0, uz0])
        elif component == "kappa":
            p0.extend([n_comp, T0 * scale, ux0, uy0, uz0, 5.0])
    return p0


def _default_bounds(vx, vy, vz, mode):
    mins = [float(np.nanmin(vx)), float(np.nanmin(vy)), float(np.nanmin(vz))]
    maxs = [float(np.nanmax(vx)), float(np.nanmax(vy)), float(np.nanmax(vz))]
    ranges = [max(hi - lo, 1.0) for lo, hi in zip(mins, maxs)]
    lower_u = [lo - r for lo, r in zip(mins, ranges)]
    upper_u = [hi + r for hi, r in zip(maxs, ranges)]

    lower = []
    upper = []
    for component in mode:
        if component == "maxwellian":
            lower.extend([1e-12, 1e-4, *lower_u])
            upper.extend([1e6, 1e7, *upper_u])
        elif component == "kappa":
            lower.extend([1e-12, 1e-4, *lower_u, 1.500001])
            upper.extend([1e6, 1e7, *upper_u, 200.0])
    return lower, upper


def _pack_x(params, mode):
    params = np.asarray(params, dtype=float)
    packed = []
    idx = 0
    for component in mode:
        if component == "maxwellian":
            packed.extend([np.log(params[idx]), np.log(params[idx + 1]),
                           params[idx + 2], params[idx + 3], params[idx + 4]])
            idx += 5
        elif component == "kappa":
            packed.extend([np.log(params[idx]), np.log(params[idx + 1]),
                           params[idx + 2], params[idx + 3], params[idx + 4],
                           np.log(params[idx + 5])])
            idx += 6
    return np.asarray(packed, dtype=float)


def _unpack_x(x, mode):
    unpacked = []
    idx = 0
    for component in mode:
        if component == "maxwellian":
            unpacked.extend([np.exp(x[idx]), np.exp(x[idx + 1]),
                             x[idx + 2], x[idx + 3], x[idx + 4]])
            idx += 5
        elif component == "kappa":
            unpacked.extend([np.exp(x[idx]), np.exp(x[idx + 1]),
                             x[idx + 2], x[idx + 3], x[idx + 4],
                             np.exp(x[idx + 5])])
            idx += 6
    return np.asarray(unpacked, dtype=float)


def _params_to_components(params, mode):
    components = []
    idx = 0
    for component in mode:
        if component == "maxwellian":
            n, T, ux, uy, uz = params[idx:idx + 5]
            components.append({
                "type": "maxwellian",
                "n": float(n),
                "T": float(T),
                "u": [float(ux), float(uy), float(uz)],
            })
            idx += 5
        elif component == "kappa":
            n, T, ux, uy, uz, kappa = params[idx:idx + 6]
            components.append({
                "type": "kappa",
                "n": float(n),
                "T": float(T),
                "u": [float(ux), float(uy), float(uz)],
                "kappa": float(kappa),
            })
            idx += 6
    return components


def _fit_one_time(vx, vy, vz, psd, species, mode, p0, bounds):
    good = (
        np.isfinite(vx)
        & np.isfinite(vy)
        & np.isfinite(vz)
        & np.isfinite(psd)
        & (psd > 0)
    )
    if np.count_nonzero(good) < 8:
        raise ValueError("Need at least eight finite positive PSD points.")

    if p0 is None:
        p0 = _initial_guess(vx, vy, vz, psd, good, species, mode)
    if bounds is None:
        bounds = _default_bounds(vx, vy, vz, mode)

    expected = sum(_component_param_count(component) for component in mode)
    if len(p0) != expected:
        raise ValueError(f"p0 has {len(p0)} values, but mode requires {expected}.")
    if len(bounds[0]) != expected or len(bounds[1]) != expected:
        raise ValueError(f"bounds must each have {expected} values.")

    def model_from_params(params):
        return _evaluate_mixture(vx, vy, vz, params, species, mode)

    log_y = np.log(psd[good])

    def residual(x):
        params = _unpack_x(x, mode)
        model = np.maximum(model_from_params(params), np.finfo(float).tiny)
        return np.log(model[good]) - log_y

    result = least_squares(
        residual,
        _pack_x(p0, mode),
        bounds=(_pack_x(bounds[0], mode), _pack_x(bounds[1], mode)),
        max_nfev=30000,
    )
    params = _unpack_x(result.x, mode)
    fit = model_from_params(params)
    fit_safe = np.maximum(fit, np.finfo(float).tiny)
    rms_log10 = float(np.sqrt(np.mean((np.log10(fit_safe[good])
                                       - np.log10(psd[good])) ** 2)))
    return {
        "mode": mode,
        "components": _params_to_components(params, mode),
        "fit": fit,
        "success": bool(result.success),
        "rms_log10": rms_log10,
    }


def fit_3d_psd(psd, vx=None, vy=None, vz=None, species=None, mode=None,
               p0=None, bounds=None):
    """
    Fit isotropic 3D Maxwellian and kappa components to PSD data.

    Parameters
    ----------
    psd : skymap-like object or ndarray
        If a skymap-like object is supplied, it must contain data, energy, phi,
        theta, and species metadata. This is the intended input after
        vdf.flux_convert(swia_3d, "def2psd"). If an ndarray is supplied, vx,
        vy, vz, and species must also be provided.
    vx, vy, vz : array-like, optional
        Velocity components in km/s. Required only when psd is an ndarray.
        Shape must match psd. Both 3D single-time and 4D time series arrays are
        accepted.
    species : str, optional
        Particle species. For skymap input, this is read from psd.attrs unless
        explicitly provided.
    mode : str or list of str, default ["maxwellian"]
        Components to fit. Supported entries are "maxwellian" and "kappa".
        Examples: ["maxwellian"], ["maxwellian", "kappa"], or
        ["kappa", "kappa"].
    p0 : list or tuple, optional
        Initial guess. Maxwellian uses (n, T, ux, uy, uz). Kappa uses
        (n, T, ux, uy, uz, kappa). Density n is in cm^-3, T is in eV, and
        velocities are in km/s.
    bounds : tuple, optional
        Lower and upper bounds in the same parameter order as p0.

    Returns
    -------
    out : dict
        For a single time slice, returns a dict with keys mode, components,
        fit, success, and rms_log10. For multiple time slices, returns a dict
        with keys mode, time, and results, where results is a list of the
        single-time output dictionaries.

    Notes
    -----
    The fitted temperature is scalar and isotropic. The optimizer is
    scipy.optimize.least_squares, applied to residuals in log(PSD).
    """

    mode = _normalize_mode(mode)
    if vx is None or vy is None or vz is None:
        psd_data, vx_data, vy_data, vz_data, species_from_data, time = _extract_from_skymap(psd)
        if species is None:
            species = species_from_data
    else:
        if species is None:
            raise ValueError("species must be provided when vx, vy, vz are provided.")
        psd_data, vx_data, vy_data, vz_data, _, time = _prepare_array_inputs(
            psd, vx, vy, vz, species
        )

    results = []
    for it in range(psd_data.shape[0]):
        results.append(_fit_one_time(
            vx_data[it],
            vy_data[it],
            vz_data[it],
            psd_data[it],
            species,
            mode,
            p0,
            bounds,
        ))

    if len(results) == 1:
        out = results[0]
        if time is not None:
            out["time"] = time[0]
        return out

    return {
        "mode": mode,
        "time": time,
        "results": results,
    }

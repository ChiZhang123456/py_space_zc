import numpy as np
from scipy.optimize import least_squares

from .get_particle_mass_charge import get_particle_mass_charge
from .maxwellian_distribution import _bi_2d as bi_maxwellian_2d
from .kappa_distribution import _bi as bi_kappa


def _speed2_to_temperature(speed2_kms2, species):
    mass, charge = get_particle_mass_charge(species)
    return mass * np.asarray(speed2_kms2, dtype=float) * 1e6 / (2.0 * abs(charge))


def _velocity_energy_ev(vpara_grid, vperp_grid, species):
    mass, charge = get_particle_mass_charge(species)
    speed2_m2s2 = (vpara_grid ** 2 + vperp_grid ** 2) * 1e6
    return mass * speed2_m2s2 / (2.0 * abs(charge))


def _energy_range_mask(energy_ev, energyrange):
    if energyrange is None:
        return np.ones_like(energy_ev, dtype=bool)

    erange = np.asarray(energyrange, dtype=float).ravel()
    if erange.size != 2:
        raise ValueError("energyrange must be None or contain two energy limits in eV.")
    emin, emax = np.nanmin(erange), np.nanmax(erange)
    return (energy_ev >= emin) & (energy_ev <= emax)


def _prepare_2d_inputs(vpara, vperp, psd, species, energyrange=None):
    vpara = np.asarray(vpara, dtype=float).ravel()
    vperp = np.asarray(vperp, dtype=float).ravel()
    psd = np.asarray(psd, dtype=float)

    expected = (vpara.size, vperp.size)
    if psd.shape != expected:
        raise ValueError(
            f"psd must have shape (len(vpara), len(vperp)) = {expected}, "
            f"but got {psd.shape}."
        )

    vpara_grid, vperp_grid = np.meshgrid(vpara, vperp, indexing="ij")
    good = (
        np.isfinite(vpara_grid)
        & np.isfinite(vperp_grid)
        & np.isfinite(psd)
        & (psd > 0)
    )
    energy_grid = _velocity_energy_ev(vpara_grid, vperp_grid, species)
    good = good & _energy_range_mask(energy_grid, energyrange)
    if np.count_nonzero(good) < 6:
        raise ValueError("Need at least six finite positive PSD points.")

    return vpara, vperp, psd, vpara_grid, vperp_grid, good


def bi_maxwellian_2d_psd_model(vpara, vperp, n, T_perp, T_para,
                               vd=0.0, species="H+"):
    """
    Evaluate a gyrotropic drifting bi-Maxwellian on a vpara-vperp grid.

    Parameters
    ----------
    vpara : array-like
        Parallel velocity in km/s.
    vperp : array-like
        Perpendicular speed in km/s.
    n : float
        Density in cm^-3.
    T_perp : float
        Perpendicular temperature in eV.
    T_para : float
        Parallel temperature in eV.
    vd : float, default 0
        Parallel drift velocity in km/s.
    species : str, default "H+"
        Particle species.

    Returns
    -------
    psd : ndarray
        PSD in s^3/m^6 with shape (len(vpara), len(vperp)).
    """

    vpara_grid, vperp_grid = np.meshgrid(vpara, vperp, indexing="ij")
    return bi_maxwellian_2d(
        n,
        T_perp,
        T_para,
        0.0,
        vd,
        vperp_grid,
        vpara_grid,
        species,
    )


def bi_kappa_2d_psd_model(vpara, vperp, n, T_perp, T_para, vd=0.0,
                          kappa=5.0, species="H+"):
    """
    Evaluate a gyrotropic drifting bi-kappa on a vpara-vperp grid.

    Parameters are the same as bi_maxwellian_2d_psd_model, with an additional
    kappa index. The kappa convention follows py_space_zc.vdf.bi_kappa.

    Returns
    -------
    psd : ndarray
        PSD in s^3/m^6 with shape (len(vpara), len(vperp)).
    """

    vpara_grid, vperp_grid = np.meshgrid(vpara, vperp, indexing="ij")
    return bi_kappa(
        n=n,
        T_perp=T_perp,
        T_parallel=T_para,
        vd=vd,
        V_perp=vperp_grid,
        V_parallel=vpara_grid,
        kappa=kappa,
        species=species,
    )


def _normalize_mode(mode):
    if mode is None:
        mode = ["bi-maxwellian"]
    if isinstance(mode, str):
        mode = [mode]

    aliases = {
        "maxwellian": "bi-maxwellian",
        "bi_maxwellian": "bi-maxwellian",
        "bi-maxwellian": "bi-maxwellian",
        "kappa": "bi-kappa",
        "bi_kappa": "bi-kappa",
        "bi-kappa": "bi-kappa",
    }
    normalized = []
    for item in mode:
        key = item.lower().strip()
        if key not in aliases:
            raise ValueError("mode entries must be 'bi-maxwellian' or 'bi-kappa'.")
        normalized.append(aliases[key])
    if len(normalized) == 0:
        raise ValueError("mode must contain at least one component.")
    return normalized


def _component_param_count(component):
    if component == "bi-maxwellian":
        return 4
    if component == "bi-kappa":
        return 5
    raise ValueError(f"Unknown component: {component}")


def _evaluate_mixture(vpara, vperp, params, species, mode):
    total = np.zeros((len(vpara), len(vperp)), dtype=float)
    idx = 0
    for component in mode:
        if component == "bi-maxwellian":
            comp_params = params[idx:idx + 4]
            total += bi_maxwellian_2d_psd_model(vpara, vperp, *comp_params,
                                                species=species)
            idx += 4
        elif component == "bi-kappa":
            comp_params = params[idx:idx + 5]
            total += bi_kappa_2d_psd_model(vpara, vperp, *comp_params,
                                           species=species)
            idx += 5
    return total


def _initial_guess(vpara_grid, vperp_grid, psd, good, species, mode):
    weights = np.asarray(psd[good], dtype=float)
    weights = weights / np.sum(weights)
    vp = vpara_grid[good]
    vt = np.abs(vperp_grid[good])

    vd0 = float(np.sum(weights * vp))
    var_para = float(np.sum(weights * (vp - vd0) ** 2))
    mean_perp2 = float(np.sum(weights * vt ** 2))
    T_para0 = max(_speed2_to_temperature(max(2.0 * var_para, 1.0), species), 1e-3)
    T_perp0 = max(_speed2_to_temperature(max(mean_perp2, 1.0), species), 1e-3)

    peak_idx = np.nanargmax(np.where(good, psd, np.nan))
    i_para, i_perp = np.unravel_index(peak_idx, psd.shape)
    unit = bi_maxwellian_2d_psd_model(
        [vpara_grid[i_para, i_perp]],
        [vperp_grid[i_para, i_perp]],
        1.0,
        T_perp0,
        T_para0,
        vd0,
        species,
    )[0, 0]
    n0 = float(psd[i_para, i_perp] / unit) if unit > 0 else 1.0
    n0 = max(n0, 1e-10)

    p0 = []
    for idx, component in enumerate(mode):
        scale = 1.0 + 0.8 * idx
        n_comp = n0 / len(mode)
        vd_comp = vd0
        if len(mode) > 1:
            vd_comp = vd0 + (idx - 0.5 * (len(mode) - 1)) * 0.2 * (
                np.nanmax(vpara_grid) - np.nanmin(vpara_grid)
            )
        if component == "bi-maxwellian":
            p0.extend([n_comp, T_perp0 * scale, T_para0 * scale, vd_comp])
        elif component == "bi-kappa":
            p0.extend([n_comp, T_perp0 * scale, T_para0 * scale, vd_comp, 5.0])
    return p0


def _default_bounds(vpara, mode):
    vmin = float(np.nanmin(vpara))
    vmax = float(np.nanmax(vpara))
    vrange = max(vmax - vmin, 1.0)
    vd_lower = vmin - vrange
    vd_upper = vmax + vrange
    lower = []
    upper = []
    for component in mode:
        if component == "bi-maxwellian":
            lower.extend([1e-12, 1e-4, 1e-4, vd_lower])
            upper.extend([1e6, 1e7, 1e7, vd_upper])
        elif component == "bi-kappa":
            lower.extend([1e-12, 1e-4, 1e-4, vd_lower, 1.500001])
            upper.extend([1e6, 1e7, 1e7, vd_upper, 200.0])
    return lower, upper


def _pack_x(params, mode):
    params = np.asarray(params, dtype=float)
    packed = []
    idx = 0
    for component in mode:
        if component == "bi-maxwellian":
            packed.extend([np.log(params[idx]), np.log(params[idx + 1]),
                           np.log(params[idx + 2]), params[idx + 3]])
            idx += 4
        elif component == "bi-kappa":
            packed.extend([np.log(params[idx]), np.log(params[idx + 1]),
                           np.log(params[idx + 2]), params[idx + 3],
                           np.log(params[idx + 4])])
            idx += 5
    return np.asarray(packed, dtype=float)


def _unpack_x(x, mode):
    unpacked = []
    idx = 0
    for component in mode:
        if component == "bi-maxwellian":
            unpacked.extend([np.exp(x[idx]), np.exp(x[idx + 1]),
                             np.exp(x[idx + 2]), x[idx + 3]])
            idx += 4
        elif component == "bi-kappa":
            unpacked.extend([np.exp(x[idx]), np.exp(x[idx + 1]),
                             np.exp(x[idx + 2]), x[idx + 3],
                             np.exp(x[idx + 4])])
            idx += 5
    return np.asarray(unpacked, dtype=float)


def _pack_bounds(bounds, mode):
    lower, upper = bounds
    return _pack_x(lower, mode), _pack_x(upper, mode)


def _params_to_components(params, mode):
    components = []
    idx = 0
    for component in mode:
        if component == "bi-maxwellian":
            n, T_perp, T_para, vd = params[idx:idx + 4]
            components.append({
                "type": "bi-maxwellian",
                "n": float(n),
                "T_perp": float(T_perp),
                "T_para": float(T_para),
                "vd": float(vd),
            })
            idx += 4
        elif component == "bi-kappa":
            n, T_perp, T_para, vd, kappa = params[idx:idx + 5]
            components.append({
                "type": "bi-kappa",
                "n": float(n),
                "T_perp": float(T_perp),
                "T_para": float(T_para),
                "vd": float(vd),
                "kappa": float(kappa),
            })
            idx += 5
    return components


def fit_2d_bi_psd(vpara, vperp, psd, species="H+", mode="bi-maxwellian",
                  p0=None, bounds=None, energyrange=None):
    """
    Fit a 2D gyrotropic bi-Maxwellian or bi-kappa PSD.

    Parameters
    ----------
    vpara : array-like
        1D parallel velocity grid in km/s.
    vperp : array-like
        1D perpendicular speed grid in km/s.
    psd : array-like
        2D PSD in s^3/m^6. Its shape must be
        (len(vpara), len(vperp)).
    species : str, default "H+"
        Particle species.
    mode : str or list of str, default "bi-maxwellian"
        Components to fit. Supported entries are "bi-maxwellian" and
        "bi-kappa". Examples: "bi-maxwellian",
        ["bi-maxwellian", "bi-kappa"], ["bi-kappa", "bi-kappa"], or any
        longer list.
    p0 : list or tuple, optional
        Initial guess. For bi-Maxwellian, use
        (n, T_perp, T_para, vd). For bi-kappa, use
        (n, T_perp, T_para, vd, kappa). Density is in cm^-3, temperatures
        are in eV, and vd is in km/s.
    bounds : tuple, optional
        Lower and upper bounds in the same parameter order as p0.
    energyrange : sequence of two floats, optional
        Total kinetic energy range in eV used for fitting. For each 2D grid
        point, energy is computed from vpara and vperp. If None, all finite
        positive PSD points are used.

    Returns
    -------
    out : dict
        Dictionary with keys:
        mode : list of str
            Fitted component types.
        components : list of dict
            Fitted parameters in the same order as mode. Each component
            contains type, n, T_perp, T_para, and vd. Bi-kappa components also
            contain kappa. Density n is in cm^-3, temperatures are in eV, and
            vd is in km/s.
        vpara : ndarray
            Validated input parallel velocity grid in km/s.
        vperp : ndarray
            Validated input perpendicular speed grid in km/s.
        fit : ndarray
            Best-fit PSD in s^3/m^6 with shape (len(vpara), len(vperp)).
        success : bool
            Whether scipy.optimize.least_squares reports convergence.
        rms_log10 : float
            Root-mean-square residual in log10(PSD).

    Notes
    -----
    The optimizer is scipy.optimize.least_squares, applied to residuals in
    log(PSD). Invalid, nonfinite, and nonpositive PSD points are ignored.
    """

    mode = _normalize_mode(mode)
    vpara, vperp, psd, vpara_grid, vperp_grid, good = _prepare_2d_inputs(
        vpara, vperp, psd, species, energyrange
    )
    if p0 is None:
        p0 = _initial_guess(vpara_grid, vperp_grid, psd, good, species, mode)
    if bounds is None:
        bounds = _default_bounds(vpara, mode)

    expected = sum(_component_param_count(component) for component in mode)
    if len(p0) != expected:
        raise ValueError(f"p0 has {len(p0)} values, but mode requires {expected}.")
    if len(bounds[0]) != expected or len(bounds[1]) != expected:
        raise ValueError(f"bounds must each have {expected} values.")

    def model_from_params(params):
        return _evaluate_mixture(vpara, vperp, params, species, mode)

    log_y = np.log(psd[good])

    def residual(x):
        params = _unpack_x(x, mode)
        model = np.maximum(model_from_params(params), np.finfo(float).tiny)
        return np.log(model[good]) - log_y

    result = least_squares(
        residual,
        _pack_x(p0, mode),
        bounds=_pack_bounds(bounds, mode),
        max_nfev=30000,
    )
    params = _unpack_x(result.x, mode)
    fit = model_from_params(params)
    rms_log10 = float(np.sqrt(np.mean((np.log10(fit[good])
                                       - np.log10(psd[good])) ** 2)))

    return {
        "mode": mode,
        "components": _params_to_components(params, mode),
        "vpara": vpara,
        "vperp": vperp,
        "fit": fit,
        "success": bool(result.success),
        "rms_log10": rms_log10,
    }

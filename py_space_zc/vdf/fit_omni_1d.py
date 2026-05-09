import numpy as np
from scipy.optimize import least_squares

from .get_particle_mass_charge import get_particle_mass_charge
from .maxwellian_distribution import _1d as maxwellian_1d
from .kappa_distribution import _1d as kappa_1d


def energy_to_speed(energy_ev, species="H+"):
    """
    Convert kinetic energy to particle speed.

    Parameters
    ----------
    energy_ev : array-like
        Particle kinetic energy in eV.
    species : str
        Particle species accepted by get_particle_mass_charge, for example
        "e", "H", "H+", "O", or "O2". Default is "H+".

    Returns
    -------
    speed_kms : ndarray
        Particle speed in km/s.
    """

    mass, charge = get_particle_mass_charge(species)
    energy_ev = np.asarray(energy_ev, dtype=float)
    return np.sqrt(2.0 * abs(charge) * energy_ev / mass) * 1e-3


def maxwellian_omni_1d_model(energy_ev, n, T, species="H+"):
    """
    Evaluate an isotropic omni reduced Maxwellian spectrum.

    Parameters
    ----------
    energy_ev : array-like
        Energy grid in eV.
    n : float
        Number density in cm^-3.
    T : float
        Temperature in eV.
    species : str, default "H+"
        Particle species.

    Returns
    -------
    f1d : ndarray
        Omni reduced 1D phase space density in s/m^4.

    Notes
    -----
    This model directly evaluates maxwellian_distribution._1d at
    V = sqrt(2*q*E/m). The returned quantity has units s/m^4.
    """

    speed_kms = energy_to_speed(energy_ev, species)
    return maxwellian_1d(n, T, 0.0, speed_kms, species)


def kappa_omni_1d_model(energy_ev, n, T, kappa, species="H+"):
    """
    Evaluate an isotropic omni reduced kappa spectrum.

    Parameters
    ----------
    energy_ev : array-like
        Energy grid in eV.
    n : float
        Number density in cm^-3.
    T : float
        Temperature in eV, using the same convention as kappa_distribution._1d.
    kappa : float
        Kappa index. Must be larger than 1.5.
    species : str, default "H+"
        Particle species.

    Returns
    -------
    f1d : ndarray
        Omni reduced 1D phase space density in s/m^4.
    """

    speed_kms = energy_to_speed(energy_ev, species)
    return kappa_1d(n, T, 0.0, speed_kms, kappa, species)


def _energy_range_mask(energy_ev, energyrange):
    if energyrange is None:
        return np.ones_like(energy_ev, dtype=bool)

    erange = np.asarray(energyrange, dtype=float).ravel()
    if erange.size != 2:
        raise ValueError("energyrange must be None or contain two energy limits in eV.")
    emin, emax = np.nanmin(erange), np.nanmax(erange)
    return (energy_ev >= emin) & (energy_ev <= emax)


def _clean_fit_arrays(energy_ev, f1d, energyrange=None):
    energy_ev = np.asarray(energy_ev, dtype=float).ravel()
    f1d = np.asarray(f1d, dtype=float).ravel()
    good = np.isfinite(energy_ev) & np.isfinite(f1d)
    good = good & (energy_ev > 0) & (f1d > 0)
    good = good & _energy_range_mask(energy_ev, energyrange)
    if np.count_nonzero(good) < 3:
        raise ValueError("Need at least three finite positive energy and f1d points.")
    return energy_ev[good], f1d[good]


def _initial_n_T_from_peak(energy_ev, f1d, species):
    peak_idx = int(np.nanargmax(f1d))
    # This is only a practical initial guess. For zero-drift 1D Maxwellian
    # spectra sampled against positive speed, f1d is monotonic in energy.
    T0 = float(np.clip(energy_ev[peak_idx], 0.1, np.inf))
    model_unit_n = maxwellian_omni_1d_model(energy_ev[peak_idx], 1.0, T0,
                                            species)
    n0 = float(f1d[peak_idx] / model_unit_n) if model_unit_n > 0 else 1.0
    return max(n0, 1e-8), max(T0, 1e-3)


def _fit_log_model(model_func, energy_ev, f1d, p0, bounds, args):
    log_y = np.log(f1d)

    def residual(log_params):
        params = np.exp(log_params)
        model = model_func(energy_ev, *params, *args)
        model = np.maximum(model, np.finfo(float).tiny)
        return np.log(model) - log_y

    log_p0 = np.log(np.asarray(p0, dtype=float))
    lower, upper = bounds
    result = least_squares(
        residual,
        log_p0,
        bounds=(np.log(lower), np.log(upper)),
        max_nfev=20000,
    )
    params = np.exp(result.x)
    fit = model_func(energy_ev, *params, *args)
    rms_log10 = float(np.sqrt(np.mean((np.log10(fit) - np.log10(f1d))**2)))
    return params, fit, result, rms_log10


def _normalize_mode(mode):
    if mode is None:
        mode = ["maxwellian"]
    if isinstance(mode, str):
        mode = [mode]

    mode = [item.lower().strip() for item in mode]
    valid = {"maxwellian", "kappa"}
    unknown = [item for item in mode if item not in valid]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Use 'maxwellian' or 'kappa'.")
    if len(mode) == 0:
        raise ValueError("mode must contain at least one component.")
    return mode


def _component_param_count(component):
    if component == "maxwellian":
        return 2
    if component == "kappa":
        return 3
    raise ValueError(f"Unknown component: {component}")


def _evaluate_mixture(energy_ev, params, species, mode):
    total = np.zeros_like(np.asarray(energy_ev, dtype=float), dtype=float)
    idx = 0
    for component in mode:
        if component == "maxwellian":
            n, T = params[idx:idx + 2]
            total += maxwellian_omni_1d_model(energy_ev, n, T, species)
            idx += 2
        elif component == "kappa":
            n, T, kappa = params[idx:idx + 3]
            total += kappa_omni_1d_model(energy_ev, n, T, kappa, species)
            idx += 3
    return total


def _guess_component_temperatures(energy_ev, mode):
    max_t = np.nanmax(energy_ev)
    min_t = np.nanmin(energy_ev)
    if len(mode) == 1:
        return [float(np.sqrt(min_t * max_t))]

    quantiles = np.linspace(0.25, 0.85, len(mode))
    temperatures = []
    for component, quantile in zip(mode, quantiles):
        if component == "kappa":
            quantile = max(quantile, 0.65)
        temperatures.append(float(np.nanquantile(energy_ev, quantile)))
    return temperatures


def _initial_mixture_guess(energy_ev, f1d, species, mode):
    temperatures = _guess_component_temperatures(energy_ev, mode)
    p0 = []
    n_floor = 1e-8
    for component, T0 in zip(mode, temperatures):
        if component == "maxwellian":
            unit_model = maxwellian_omni_1d_model(energy_ev[0], 1.0, T0, species)
            n0 = float(f1d[0] / unit_model) if unit_model > 0 else 1.0
            p0.extend([max(n0 / len(mode), n_floor), max(T0, 1e-3)])
        elif component == "kappa":
            high_idx = -1
            unit_model = kappa_omni_1d_model(energy_ev[high_idx], 1.0, T0,
                                             5.0, species)
            n0 = float(f1d[high_idx] / unit_model) if unit_model > 0 else 1.0
            p0.extend([max(n0 / len(mode), n_floor), max(T0, 1e-3), 5.0])
    return p0


def _default_mixture_bounds(mode):
    lower = []
    upper = []
    for component in mode:
        if component == "maxwellian":
            lower.extend([1e-12, 1e-4])
            upper.extend([1e6, 1e7])
        elif component == "kappa":
            lower.extend([1e-12, 1e-4, 1.500001])
            upper.extend([1e6, 1e7, 200.0])
    return lower, upper


def _params_to_components(params, mode):
    components = []
    idx = 0
    for component in mode:
        if component == "maxwellian":
            n, T = params[idx:idx + 2]
            components.append({
                "type": "maxwellian",
                "n": float(n),
                "T": float(T),
            })
            idx += 2
        elif component == "kappa":
            n, T, kappa = params[idx:idx + 3]
            components.append({
                "type": "kappa",
                "n": float(n),
                "T": float(T),
                "kappa": float(kappa),
            })
            idx += 3
    return components


def fit_omni_1d(energy_ev, psd_1d, species="H+", mode=None, p0=None,
                bounds=None, energyrange=None):
    """
    Fit one or more 1D Maxwellian and kappa components to an energy spectrum.

    Parameters
    ----------
    energy_ev : array-like
        Energy grid in eV.
    psd_1d : array-like
        Input 1D reduced PSD in s/m^4.
    species : str, default "H+"
        Particle species.
    mode : list of str, default ["maxwellian"]
        Components to fit. Supported entries are "maxwellian" and "kappa".
        Examples: ["maxwellian"], ["maxwellian", "kappa"],
        ["maxwellian", "kappa", "kappa"], or
        ["maxwellian", "maxwellian"].
    p0 : list or tuple, optional
        Initial guess. Parameter order follows mode. Maxwellian uses (n, T).
        Kappa uses (n, T, kappa). Density is in cm^-3, T is in eV.
    bounds : tuple, optional
        Lower and upper bounds in the same parameter order as p0.
    energyrange : sequence of two floats, optional
        Energy range in eV used for fitting. If None, all finite positive
        points are used.

    Returns
    -------
    out : dict
        Dictionary with these keys:
        mode : list of str
            Component types used in the fit.
        components : list of dict
            Fitted component parameters in the same order as mode. A
            Maxwellian component has type, n, and T. A kappa component has
            type, n, T, and kappa. Density n is in cm^-3 and T is in eV.
        energy : ndarray
            Energy points used in the fit, after removing invalid values.
        fit : ndarray
            Best-fit total PSD in s/m^4.
        success : bool
            Whether scipy.optimize.least_squares reports convergence.
        rms_log10 : float
            Root-mean-square residual in log10(PSD).

    Notes
    -----
    The total model is a linear sum of all component PSDs. The optimizer is
    scipy.optimize.least_squares, applied to residuals in log(PSD).
    """

    mode = _normalize_mode(mode)
    energy_ev, psd_1d = _clean_fit_arrays(energy_ev, psd_1d, energyrange)

    if p0 is None:
        p0 = _initial_mixture_guess(energy_ev, psd_1d, species, mode)
    if bounds is None:
        bounds = _default_mixture_bounds(mode)

    expected = sum(_component_param_count(component) for component in mode)
    if len(p0) != expected:
        raise ValueError(f"p0 has {len(p0)} values, but mode requires {expected}.")
    if len(bounds[0]) != expected or len(bounds[1]) != expected:
        raise ValueError(f"bounds must each have {expected} values.")

    params, fit, result, rms_log10 = _fit_log_model(
        lambda e, *pars: _evaluate_mixture(e, pars, species, mode),
        energy_ev,
        psd_1d,
        p0,
        bounds,
        (),
    )

    return {
        "mode": mode,
        "components": _params_to_components(params, mode),
        "energy": energy_ev,
        "fit": fit,
        "success": bool(result.success),
        "rms_log10": rms_log10,
    }


def fit_maxwellian_omni_1d(energy_ev, psd_1d, species="H+", p0=None,
                           bounds=None, energyrange=None):
    """
    Fit an isotropic omni reduced Maxwellian to a 1D energy spectrum.

    Parameters
    ----------
    energy_ev : array-like
        Energy grid in eV.
    psd_1d : array-like
        Input 1D reduced PSD in s/m^4.
    species : str, default "H+"
        Particle species.
    p0 : tuple or list, optional
        Initial guess (n, T), where n is in cm^-3 and T is in eV.
    bounds : tuple, optional
        Lower and upper bounds for (n, T). Defaults to
        ([1e-12, 1e-4], [1e6, 1e7]).
    energyrange : sequence of two floats, optional
        Energy range in eV used for fitting. If None, all finite positive
        points are used.

    Returns
    -------
    out : dict
        Same keys as fit_omni_1d, plus n and T for convenient access.
        The fit array has units s/m^4.

    Notes
    -----
    This function uses scipy.optimize.least_squares and minimizes residuals in
    log(PSD), not linear PSD.
    """

    out = fit_omni_1d(
        energy_ev, psd_1d, species, ["maxwellian"], p0, bounds, energyrange
    )
    n = out["components"][0]["n"]
    T = out["components"][0]["T"]
    return {
        **out,
        "n": n,
        "T": T,
    }


def fit_kappa_omni_1d(energy_ev, psd_1d, species="H+", p0=None,
                      bounds=None, energyrange=None):
    """
    Fit an isotropic omni reduced kappa distribution to a 1D energy spectrum.

    Parameters
    ----------
    energy_ev : array-like
        Energy grid in eV.
    psd_1d : array-like
        Input 1D reduced PSD in s/m^4.
    species : str, default "H+"
        Particle species.
    p0 : tuple or list, optional
        Initial guess (n, T, kappa), where n is in cm^-3 and T is in eV.
    bounds : tuple, optional
        Lower and upper bounds for (n, T, kappa). Defaults to
        ([1e-12, 1e-4, 1.500001], [1e6, 1e7, 200]).
    energyrange : sequence of two floats, optional
        Energy range in eV used for fitting. If None, all finite positive
        points are used.

    Returns
    -------
    out : dict
        Same keys as fit_omni_1d, plus n, T, and kappa for convenient access.
        The fit array has units s/m^4.

    Notes
    -----
    This function uses scipy.optimize.least_squares and minimizes residuals in
    log(PSD), not linear PSD.
    """

    out = fit_omni_1d(
        energy_ev, psd_1d, species, ["kappa"], p0, bounds, energyrange
    )
    n = out["components"][0]["n"]
    T = out["components"][0]["T"]
    kappa = out["components"][0]["kappa"]
    return {
        **out,
        "n": n,
        "T": T,
        "kappa": kappa,
    }

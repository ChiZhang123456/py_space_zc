import numpy as np
from scipy.optimize import least_squares

from .get_particle_mass_charge import get_particle_mass_charge
from .kappa_distribution import _bi as bi_kappa


def energy_to_speed_kms(energy_ev, species="H+"):
    """Convert kinetic energy in eV to particle speed in km/s."""

    mass, charge = get_particle_mass_charge(species)
    return np.sqrt(2.0 * abs(charge) * np.asarray(energy_ev, dtype=float) / mass) * 1e-3


def energy_pitchangle_to_vpara_vperp(energy_ev, pitchangle_deg, species="H+"):
    """
    Convert energy and pitch angle grids to parallel and perpendicular velocity.

    Parameters
    ----------
    energy_ev : array-like
        One-dimensional energy grid in eV.
    pitchangle_deg : array-like
        One-dimensional pitch angle grid in degrees.
    species : str
        Particle species accepted by get_particle_mass_charge.

    Returns
    -------
    vpara, vperp : ndarray
        Two-dimensional arrays with shape
        (len(energy_ev), len(pitchangle_deg)), in km/s.
    """

    energy_grid, pa_grid = np.meshgrid(
        np.asarray(energy_ev, dtype=float),
        np.asarray(pitchangle_deg, dtype=float),
        indexing="ij",
    )
    speed = energy_to_speed_kms(energy_grid, species)
    pitch_rad = np.deg2rad(pa_grid)
    vpara = speed * np.cos(pitch_rad)
    vperp = speed * np.sin(pitch_rad)
    return vpara, vperp


def bi_kappa_energy_pa_model(energy_ev, pitchangle_deg, params, species="H+"):
    """
    Evaluate one or more drifting bi-kappa distributions on an energy-PA grid.

    Parameters
    ----------
    energy_ev : array-like
        One-dimensional energy grid in eV.
    pitchangle_deg : array-like
        One-dimensional pitch angle grid in degrees.
    params : sequence
        Flat parameter list. Each component uses
        (n, T_perp, T_para, vd, kappa), where n is in cm^-3, temperatures
        are in eV, vd is in km/s, and kappa > 1.5.
    species : str
        Particle species.

    Returns
    -------
    psd : ndarray
        Full 3D gyrotropic phase space density in s^3/m^6 with shape
        (len(energy_ev), len(pitchangle_deg)).
    """

    params = np.asarray(params, dtype=float).ravel()
    if params.size % 5 != 0:
        raise ValueError("params must contain 5 values per bi-kappa component.")

    vpara, vperp = energy_pitchangle_to_vpara_vperp(
        energy_ev, pitchangle_deg, species
    )
    model = np.zeros_like(vpara, dtype=float)
    for i in range(params.size // 5):
        n, T_perp, T_para, vd, kappa = params[5 * i:5 * (i + 1)]
        model += bi_kappa(
            n=n,
            T_perp=T_perp,
            T_parallel=T_para,
            vd=vd,
            V_perp=vperp,
            V_parallel=vpara,
            kappa=kappa,
            species=species,
        )
    return model


def _energy_range_mask(energy_ev, pitchangle_deg, energyrange):
    energy_grid, _ = np.meshgrid(
        np.asarray(energy_ev, dtype=float),
        np.asarray(pitchangle_deg, dtype=float),
        indexing="ij",
    )
    if energyrange is None:
        return np.ones_like(energy_grid, dtype=bool)

    erange = np.asarray(energyrange, dtype=float).ravel()
    if erange.size != 2:
        raise ValueError("energyrange must be None or contain two limits in eV.")
    emin, emax = np.nanmin(erange), np.nanmax(erange)
    return (energy_grid >= emin) & (energy_grid <= emax)


def _speed2_to_temperature(speed2_kms2, species):
    mass, charge = get_particle_mass_charge(species)
    return mass * np.asarray(speed2_kms2, dtype=float) * 1e6 / (2.0 * abs(charge))


def _prepare_inputs(energy_ev, pitchangle_deg, psd, species, energyrange):
    energy_ev = np.asarray(energy_ev, dtype=float).ravel()
    pitchangle_deg = np.asarray(pitchangle_deg, dtype=float).ravel()
    psd = np.asarray(psd, dtype=float)

    expected = (energy_ev.size, pitchangle_deg.size)
    if psd.shape != expected:
        raise ValueError(
            f"psd must have shape (len(energy), len(pitchangle)) = {expected}, "
            f"but got {psd.shape}."
        )

    vpara, vperp = energy_pitchangle_to_vpara_vperp(
        energy_ev, pitchangle_deg, species
    )
    good = (
        np.isfinite(vpara)
        & np.isfinite(vperp)
        & np.isfinite(psd)
        & (psd > 0)
        & _energy_range_mask(energy_ev, pitchangle_deg, energyrange)
    )
    if np.count_nonzero(good) < 10:
        raise ValueError("Need at least ten finite positive PSD points for fitting.")
    return energy_ev, pitchangle_deg, psd, vpara, vperp, good


def _initial_guess(vpara, vperp, psd, good, species, n_components):
    weights = np.asarray(psd[good], dtype=float)
    weights = weights / np.sum(weights)
    vp = vpara[good]
    vt = np.abs(vperp[good])

    vd0 = float(np.sum(weights * vp))
    var_para = float(np.sum(weights * (vp - vd0) ** 2))
    mean_perp2 = float(np.sum(weights * vt ** 2))
    T_para0 = max(float(_speed2_to_temperature(max(2.0 * var_para, 1.0), species)), 1e-3)
    T_perp0 = max(float(_speed2_to_temperature(max(mean_perp2, 1.0), species)), 1e-3)

    peak_flat = np.nanargmax(np.where(good, psd, np.nan))
    peak_idx = np.unravel_index(peak_flat, psd.shape)
    unit = bi_kappa(
        n=1.0,
        T_perp=T_perp0,
        T_parallel=T_para0,
        vd=vd0,
        V_perp=vperp[peak_idx],
        V_parallel=vpara[peak_idx],
        kappa=5.0,
        species=species,
    )
    n0 = float(psd[peak_idx] / unit) if np.isfinite(unit) and unit > 0 else 1.0
    n0 = max(n0, 1e-10)

    vspan = max(float(np.nanmax(vpara) - np.nanmin(vpara)), 1.0)
    p0 = []
    for i in range(n_components):
        scale = 1.0 + 0.9 * i
        vd = vd0
        if n_components > 1:
            vd = vd0 + (i - 0.5 * (n_components - 1)) * 0.25 * vspan
        p0.extend([n0 / n_components, T_perp0 * scale, T_para0 * scale, vd, 5.0])
    return np.asarray(p0, dtype=float)


def _default_bounds(vpara, n_components):
    vmin = float(np.nanmin(vpara))
    vmax = float(np.nanmax(vpara))
    vspan = max(vmax - vmin, 1.0)
    vd_lower = vmin - vspan
    vd_upper = vmax + vspan

    lower = []
    upper = []
    for _ in range(n_components):
        lower.extend([1e-12, 1e-4, 1e-4, vd_lower, 1.500001])
        upper.extend([1e6, 1e7, 1e7, vd_upper, 200.0])
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _pack_params(params):
    params = np.asarray(params, dtype=float)
    x = []
    for i in range(params.size // 5):
        n, T_perp, T_para, vd, kappa = params[5 * i:5 * (i + 1)]
        x.extend([np.log(n), np.log(T_perp), np.log(T_para), vd, np.log(kappa - 1.5)])
    return np.asarray(x, dtype=float)


def _unpack_params(x):
    x = np.asarray(x, dtype=float)
    params = []
    for i in range(x.size // 5):
        ln_n, ln_Tp, ln_Ta, vd, ln_km32 = x[5 * i:5 * (i + 1)]
        params.extend([
            np.exp(ln_n),
            np.exp(ln_Tp),
            np.exp(ln_Ta),
            vd,
            1.5 + np.exp(ln_km32),
        ])
    return np.asarray(params, dtype=float)


def _params_to_components(params):
    components = []
    for i in range(len(params) // 5):
        n, T_perp, T_para, vd, kappa = params[5 * i:5 * (i + 1)]
        components.append({
            "type": "bi-kappa",
            "n": float(n),
            "T_perp": float(T_perp),
            "T_para": float(T_para),
            "vd": float(vd),
            "kappa": float(kappa),
        })
    return components


def fit_bi_kappa_energy_pa(
    energy_ev,
    pitchangle_deg,
    psd,
    species="H+",
    energyrange=None,
    n_components=1,
    p0=None,
    bounds=None,
    max_nfev=50000,
):
    """
    Fit one or more drifting bi-kappa distributions to PSD(E, pitch angle).

    Parameters
    ----------
    energy_ev : array-like
        One-dimensional energy grid in eV.
    pitchangle_deg : array-like
        One-dimensional pitch angle grid in degrees.
    psd : array-like
        PSD in s^3/m^6 with shape (len(energy_ev), len(pitchangle_deg)).
    species : str, default "H+"
        Particle species.
    energyrange : sequence of two floats, optional
        Energy range in eV used for fitting. If None, all valid energies are used.
    n_components : int, default 1
        Number of bi-kappa components to fit.
    p0 : sequence, optional
        Initial guess. Each component uses
        (n, T_perp, T_para, vd, kappa), where n is in cm^-3, temperatures
        are in eV, vd is in km/s, and kappa is dimensionless.
    bounds : tuple, optional
        Lower and upper bounds in the same flat parameter order as p0.
    max_nfev : int, default 50000
        Maximum number of residual evaluations passed to scipy.optimize.least_squares.

    Returns
    -------
    out : dict
        Dictionary containing fitted component parameters, the total best-fit
        PSD, individual component PSDs, velocity grids, fit mask, and residual
        metrics. The residual metric rms_log10 is computed in log10(PSD).
    """

    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    energy_ev, pitchangle_deg, psd, vpara, vperp, good = _prepare_inputs(
        energy_ev, pitchangle_deg, psd, species, energyrange
    )
    if p0 is None:
        p0 = _initial_guess(vpara, vperp, psd, good, species, n_components)
    else:
        p0 = np.asarray(p0, dtype=float).ravel()

    expected = 5 * n_components
    if p0.size != expected:
        raise ValueError(f"p0 has {p0.size} values, but n_components requires {expected}.")

    if bounds is None:
        bounds = _default_bounds(vpara, n_components)
    lower, upper = [np.asarray(item, dtype=float).ravel() for item in bounds]
    if lower.size != expected or upper.size != expected:
        raise ValueError(f"bounds must each contain {expected} values.")

    margin = 1e-12 * np.maximum(upper - lower, 1.0)
    p0 = np.minimum(np.maximum(p0, lower + margin), upper - margin)
    log_y = np.log(psd[good])

    def model_from_params(params):
        return bi_kappa_energy_pa_model(energy_ev, pitchangle_deg, params, species)

    def residual(x):
        params = _unpack_params(x)
        model = np.maximum(model_from_params(params), np.finfo(float).tiny)
        return np.log(model[good]) - log_y

    result = least_squares(
        residual,
        _pack_params(p0),
        bounds=(_pack_params(lower), _pack_params(upper)),
        max_nfev=max_nfev,
    )

    params = _unpack_params(result.x)
    fit = model_from_params(params)
    component_fits = []
    for i in range(n_components):
        component_fits.append(
            bi_kappa_energy_pa_model(
                energy_ev, pitchangle_deg, params[5 * i:5 * (i + 1)], species
            )
        )

    rms_log10 = float(np.sqrt(np.mean((np.log10(fit[good]) - np.log10(psd[good])) ** 2)))
    return {
        "success": bool(result.success),
        "message": result.message,
        "n_components": n_components,
        "components": _params_to_components(params),
        "params": params,
        "energy": energy_ev,
        "pitchangle": pitchangle_deg,
        "vpara": vpara,
        "vperp": vperp,
        "fit": fit,
        "component_fits": component_fits,
        "fit_mask": good,
        "rms_log10": rms_log10,
        "cost": float(result.cost),
    }


def plot_bi_kappa_fit(fit_result, psd, filename=None, title=None):
    """Plot observed PSD, fitted PSD, and fit/data ratio."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    energy = fit_result["energy"]
    pitchangle = fit_result["pitchangle"]
    fit = fit_result["fit"]
    psd = np.asarray(psd, dtype=float)

    positive = psd[np.isfinite(psd) & (psd > 0)]
    if positive.size == 0:
        raise ValueError("psd must contain at least one finite positive value.")
    vmin = max(float(np.nanpercentile(positive, 2)), float(np.nanmax(positive)) * 1e-8)
    vmax = float(np.nanmax(positive))

    ratio = np.full_like(psd, np.nan, dtype=float)
    good = np.isfinite(psd) & (psd > 0)
    ratio[good] = fit[good] / psd[good]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    panels = [
        (psd, "PSD"),
        (fit, "Fit"),
        (ratio, "Fit / PSD"),
    ]
    for ax, (data, label) in zip(axes, panels):
        if label == "Fit / PSD":
            mesh = ax.pcolormesh(
                pitchangle,
                energy,
                data,
                shading="auto",
                cmap="coolwarm",
                vmin=0.7,
                vmax=1.3,
            )
        else:
            mesh = ax.pcolormesh(
                pitchangle,
                energy,
                data,
                shading="auto",
                cmap="viridis",
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
        ax.set_yscale("log")
        ax.set_xlabel("Pitch angle (deg)")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax)

    if title is None:
        title = (
            f"{fit_result['n_components']} bi-kappa component(s), "
            f"rms log10 = {fit_result['rms_log10']:.3g}"
        )
    fig.suptitle(title)
    if filename is not None:
        fig.savefig(filename, dpi=200)
        plt.close(fig)
    return fig, axes


fit_bi_kappa = fit_bi_kappa_energy_pa


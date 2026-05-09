#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Proton beta-anisotropy instability thresholds.

This module provides two small utilities for the standard space-physics
``beta_parallel`` versus ``T_perp / T_parallel`` diagram:

1. ``plot_beta_anisotropy_instability`` draws the common proton instability
   threshold curves.
2. ``check_beta_anisotropy_instability`` compares data points with the
   threshold curves and reports which modes may be unstable.

The default coefficients are the commonly used Hellinger et al. (2006)
empirical fits for maximum growth rate gamma = 1e-2 Omega_p:

    R = T_perp / T_parallel = 1 + a / (beta_parallel - beta0)**b

where ``R`` is the proton temperature anisotropy.

Reference
---------
Hellinger, P., Travnicek, P., Kasper, J. C., and Lazarus, A. J. (2006),
Solar wind proton temperature anisotropy: Linear theory and WIND/SWE
observations, Geophysical Research Letters, 33, L09101.
"""

from __future__ import annotations

import numpy as np


HELLINGER_2006_COEFFICIENTS = {
    1e-2: {
        "mirror": {
            "label": "Mirror",
            "a": 0.77,
            "b": 0.76,
            "beta0": -0.016,
            "color": "#1f77b4",
            "linestyle": "-",
            "side": "upper",
        },
        "proton_cyclotron": {
            "label": "Proton cyclotron",
            "a": 0.43,
            "b": 0.42,
            "beta0": 0.0004,
            "color": "#d62728",
            "linestyle": "--",
            "side": "upper",
        },
        "oblique_firehose": {
            "label": "Oblique firehose",
            "a": -1.40,
            "b": 1.00,
            "beta0": -0.11,
            "color": "#2ca02c",
            "linestyle": "-",
            "side": "lower",
        },
        "parallel_firehose": {
            "label": "Parallel firehose",
            "a": -0.47,
            "b": 0.53,
            "beta0": 0.59,
            "color": "#9467bd",
            "linestyle": "--",
            "side": "lower",
        },
    }
}


def _get_coefficients(gamma):
    """Return fitting coefficients for a supported maximum growth rate."""
    for gamma_key, coefficients in HELLINGER_2006_COEFFICIENTS.items():
        if np.isclose(float(gamma), gamma_key, rtol=0.0, atol=gamma_key * 1e-8):
            return coefficients

    supported = sorted(HELLINGER_2006_COEFFICIENTS)
    raise ValueError(
        f"Unsupported gamma={gamma}. Available fitted gamma values are {supported}. "
        "Add another coefficient table before using a different gamma."
    )


def _threshold(beta_parallel, a, b, beta0):
    """Evaluate one anisotropy threshold curve."""
    beta_parallel = np.asarray(beta_parallel, dtype=float)
    out = np.full(beta_parallel.shape, np.nan, dtype=float)
    good = np.isfinite(beta_parallel) & (beta_parallel > beta0)
    out[good] = 1.0 + a / np.power(beta_parallel[good] - beta0, b)
    out[out <= 0.0] = np.nan
    return out


def _broadcast_plasma_arrays(T_parallel, T_perp, beta_parallel, beta_perp):
    """Convert inputs to broadcast-compatible float arrays."""
    arrays = np.broadcast_arrays(
        np.asarray(T_parallel, dtype=float),
        np.asarray(T_perp, dtype=float),
        np.asarray(beta_parallel, dtype=float),
        np.asarray(beta_perp, dtype=float),
    )
    return arrays


def _combined_curve(curves, mode):
    """Combine threshold curves while ignoring all-NaN columns."""
    stack = np.vstack([np.ravel(curve) for curve in curves])
    valid = np.any(np.isfinite(stack), axis=0)
    out = np.full(stack.shape[1], np.nan)

    if mode == "min":
        out[valid] = np.nanmin(stack[:, valid], axis=0)
    elif mode == "max":
        out[valid] = np.nanmax(stack[:, valid], axis=0)
    else:
        raise ValueError("mode must be 'min' or 'max'")

    return out.reshape(np.asarray(curves[0]).shape)


def plot_beta_anisotropy_instability(
    T_parallel=None,
    T_perp=None,
    beta_parallel=None,
    beta_perp=None,
    gamma=1e-2,
    beta_range=(1e-2, 1e2),
    anisotropy_range=(1e-1, 1e1),
    ax=None,
    show_unstable_regions=True,
    scatter_kwargs=None,
):
    """Plot proton beta-anisotropy instability thresholds.

    Parameters
    ----------
    T_parallel, T_perp : array_like, optional
        Parallel and perpendicular temperatures. They can be any consistent
        unit because only the ratio ``T_perp / T_parallel`` is used. If these
        arrays are provided together with ``beta_parallel``, the data points are
        overplotted on the threshold diagram.
    beta_parallel, beta_perp : array_like, optional
        Parallel and perpendicular plasma beta. ``beta_parallel`` is used as
        the x-coordinate. ``beta_perp`` is accepted for a symmetric calling
        interface and for consistency checks by the user, but the plotted
        anisotropy is computed from the temperature ratio.
    gamma : float, optional
        Maximum growth rate in units of the proton gyrofrequency. The default
        is 1e-2. Different gamma values require different fitting
        coefficients. Currently, only gamma=1e-2 is included.
    beta_range : tuple, optional
        Minimum and maximum ``beta_parallel`` for the x-axis.
    anisotropy_range : tuple, optional
        Minimum and maximum ``T_perp / T_parallel`` for the y-axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If omitted, a new figure and axes are created.
    show_unstable_regions : bool, optional
        If True, lightly shade the nominal unstable sides of the thresholds.
    scatter_kwargs : dict, optional
        Keyword arguments passed to ``Axes.scatter`` for optional data points.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Figure and axes containing the instability threshold diagram.
    """
    import matplotlib.pyplot as plt

    coefficients = _get_coefficients(gamma)
    beta_grid = np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), 1200)
    curves = {
        name: _threshold(beta_grid, pars["a"], pars["b"], pars["beta0"])
        for name, pars in coefficients.items()
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    else:
        fig = ax.figure

    if show_unstable_regions:
        upper = _combined_curve(
            [curves[name] for name, pars in coefficients.items() if pars["side"] == "upper"],
            mode="min",
        )
        lower = _combined_curve(
            [curves[name] for name, pars in coefficients.items() if pars["side"] == "lower"],
            mode="max",
        )

        ax.fill_between(
            beta_grid,
            upper,
            anisotropy_range[1],
            where=np.isfinite(upper) & (upper < anisotropy_range[1]),
            color="#e8eef7",
            alpha=0.75,
            lw=0,
            label=r"$T_{\perp p} > T_{\parallel p}$ unstable side",
        )
        ax.fill_between(
            beta_grid,
            anisotropy_range[0],
            lower,
            where=np.isfinite(lower) & (lower > anisotropy_range[0]),
            color="#f3eadf",
            alpha=0.75,
            lw=0,
            label=r"$T_{\parallel p} > T_{\perp p}$ unstable side",
        )

    for name, pars in coefficients.items():
        ax.plot(
            beta_grid,
            curves[name],
            color=pars["color"],
            linestyle=pars["linestyle"],
            lw=2.2,
            label=pars["label"],
        )

    ax.axhline(1.0, color="0.25", lw=1.0, linestyle=":", label="Isotropy")

    if T_parallel is not None or T_perp is not None or beta_parallel is not None or beta_perp is not None:
        if T_parallel is None or T_perp is None or beta_parallel is None or beta_perp is None:
            raise ValueError(
                "To overplot data points, provide T_parallel, T_perp, beta_parallel, and beta_perp."
            )

        T_parallel, T_perp, beta_parallel, beta_perp = _broadcast_plasma_arrays(
            T_parallel, T_perp, beta_parallel, beta_perp
        )
        anisotropy = T_perp / T_parallel
        good = (
            np.isfinite(beta_parallel)
            & np.isfinite(anisotropy)
            & (beta_parallel > 0.0)
            & (anisotropy > 0.0)
        )

        scatter_style = {
            "s": 18,
            "c": "k",
            "alpha": 0.55,
            "linewidths": 0.0,
            "label": "Input plasma state",
            "zorder": 5,
        }
        if scatter_kwargs is not None:
            scatter_style.update(scatter_kwargs)

        ax.scatter(beta_parallel[good], anisotropy[good], **scatter_style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(beta_range)
    ax.set_ylim(anisotropy_range)
    ax.set_xlabel(r"$\beta_{\parallel p}$")
    ax.set_ylabel(r"$R_p = T_{\perp p}/T_{\parallel p}$")
    ax.set_title("Proton Temperature Anisotropy Instability Thresholds")
    ax.grid(True, which="major", color="0.82", lw=0.8)
    ax.grid(True, which="minor", color="0.90", lw=0.45)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92)

    return fig, ax


def check_beta_anisotropy_instability(
    T_parallel,
    T_perp,
    beta_parallel,
    beta_perp,
    gamma=1e-2,
):
    """Check proton anisotropy-driven instability thresholds.

    Parameters
    ----------
    T_parallel, T_perp : array_like
        Parallel and perpendicular temperatures. Inputs may be scalar-like,
        shape ``(1,)``, shape ``(n, 1)``, shape ``(1, 1)``, or any mutually
        broadcast-compatible NumPy array shape.
    beta_parallel, beta_perp : array_like
        Parallel and perpendicular plasma beta. ``beta_parallel`` is used in
        the threshold fits. ``beta_perp`` is returned for reference because it
        is often useful when checking data consistency.
    gamma : float, optional
        Maximum growth rate in units of the proton gyrofrequency. The default
        is 1e-2. Different gamma values require different fitting
        coefficients. Currently, only gamma=1e-2 is included.

    Returns
    -------
    result : dict
        Dictionary with the broadcast input arrays, the temperature anisotropy,
        threshold values for each instability, boolean instability flags, and a
        text label for the likely unstable mode at each point.

    Notes
    -----
    For ``T_perp / T_parallel > 1``, a point is flagged as mirror or proton
    cyclotron unstable when its anisotropy is above the corresponding upper
    threshold. For ``T_perp / T_parallel < 1``, a point is flagged as firehose
    unstable when its anisotropy is below the corresponding lower threshold.
    These are threshold checks, not a full dispersion-solver calculation.
    """
    coefficients = _get_coefficients(gamma)
    T_parallel, T_perp, beta_parallel, beta_perp = _broadcast_plasma_arrays(
        T_parallel, T_perp, beta_parallel, beta_perp
    )
    anisotropy = T_perp / T_parallel

    thresholds = {
        name: _threshold(beta_parallel, pars["a"], pars["b"], pars["beta0"])
        for name, pars in coefficients.items()
    }

    unstable = {}
    margins = {}
    for name, pars in coefficients.items():
        curve = thresholds[name]
        valid = np.isfinite(anisotropy) & np.isfinite(curve) & (anisotropy > 0.0)

        if pars["side"] == "upper":
            unstable[name] = valid & (anisotropy >= curve)
            margins[name] = anisotropy / curve - 1.0
        else:
            unstable[name] = valid & (anisotropy <= curve)
            margins[name] = curve / anisotropy - 1.0

        margins[name] = np.where(valid, margins[name], np.nan)

    likely_mode = np.full(anisotropy.shape, "stable", dtype=object)
    mode_order = ["mirror", "proton_cyclotron", "oblique_firehose", "parallel_firehose"]
    for index in np.ndindex(anisotropy.shape):
        active = [name for name in mode_order if unstable[name][index]]
        if active:
            likely_mode[index] = " + ".join(active)
        elif not (
            np.isfinite(T_parallel[index])
            and np.isfinite(T_perp[index])
            and np.isfinite(beta_parallel[index])
            and np.isfinite(beta_perp[index])
            and T_parallel[index] > 0.0
            and T_perp[index] > 0.0
            and beta_parallel[index] > 0.0
            and beta_perp[index] > 0.0
        ):
            likely_mode[index] = "invalid"

    return {
        "T_parallel": T_parallel,
        "T_perp": T_perp,
        "beta_parallel": beta_parallel,
        "beta_perp": beta_perp,
        "anisotropy": anisotropy,
        "thresholds": thresholds,
        "unstable": unstable,
        "margin": margins,
        "likely_mode": likely_mode,
        "gamma": gamma,
        "coefficients": coefficients,
    }


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    plot_beta_anisotropy_instability()
    plt.show()
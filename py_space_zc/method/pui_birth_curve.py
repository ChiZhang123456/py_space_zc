#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pickup-ion birth curve in uniform upstream electric and magnetic fields.

This module is a Python version of ``birth_curve.m``.

Applicability
-------------
Use this function only when the production condition is fully steady and when
the IMF and solar-wind velocity are constant over the traced gyromotion time.
The model assumes uniform fields and ions created at rest in the planet frame,

    dv/dt = (q/m) * (E + v x B), with E = -Usw x B.

The birth-curve calculation itself is planet independent. Exosphere or
production-rate profiles should be handled outside this function, so the same
trajectory calculation can be used for Mars, Mercury, or other bodies.

Example
-------
>>> from py_space_zc.method import pui_birth_curve
>>> curve = pui_birth_curve(
...     Usw=[-400, 0, 0],     # km/s
...     IMF=[0, 3, 0],        # nT
...     Pobs=[4000, 0, 0],    # km
... )
>>> curve["H"]["Pbirth"].shape
(800001, 3)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ELEMENTARY_CHARGE_C = 1.602176634e-19
PROTON_MASS_KG = 1.67262192369e-27


@dataclass(frozen=True)
class PuiSpecies:
    """Particle mass and charge for the analytic birth-curve solution."""

    mass_kg: float
    charge_c: float = ELEMENTARY_CHARGE_C


_SPECIES = {
    "H": PuiSpecies(PROTON_MASS_KG),
    "O": PuiSpecies(16.0 * PROTON_MASS_KG),
}


def pui_birth_curve(
    Usw,
    IMF,
    Pobs,
    n_gyro_h: float = 16,
    n_gyro_o: float = 1,
    n_step_per_gyro: int = 50000,
):
    """Compute H+ and O+ pickup-ion birth curves.

    Parameters
    ----------
    Usw : array_like, shape (3,)
        Solar-wind velocity vector in km/s.
    IMF : array_like, shape (3,)
        Magnetic field vector in nT.
    Pobs : array_like, shape (3,)
        Detector or sample position in km, in the same coordinates as ``Usw``
        and ``IMF``.
    n_gyro_h, n_gyro_o : float, optional
        Number of H+ and O+ gyroperiods to trace.
    n_step_per_gyro : int, optional
        Number of sampling steps per gyroperiod.

    Returns
    -------
    dict
        ``{"H": h_curve, "O": o_curve}``. Each curve contains ``T`` in s,
        ``Pbirth`` in km, ``Vdet`` in km/s, and ``Eev`` in eV.
    """

    return {
        "H": pui_birth_curve_one_species(
            Usw, IMF, Pobs, "H", n_gyro=n_gyro_h, n_step_per_gyro=n_step_per_gyro
        ),
        "O": pui_birth_curve_one_species(
            Usw, IMF, Pobs, "O", n_gyro=n_gyro_o, n_step_per_gyro=n_step_per_gyro
        ),
    }


def pui_birth_curve_one_species(
    Usw,
    IMF,
    Pobs,
    species: str | None = "H",
    n_gyro: float = 1,
    n_step_per_gyro: int = 50000,
    mass_kg: float | None = None,
    charge_c: float = ELEMENTARY_CHARGE_C,
):
    """Compute a pickup-ion birth curve for one species.

    Parameters
    ----------
    Usw, IMF, Pobs : array_like, shape (3,)
        Solar-wind velocity in km/s, magnetic field in nT, and observation
        position in km.
    species : {"H", "O", None}, optional
        Built-in species name. Ignored when ``mass_kg`` is supplied.
    n_gyro : float, optional
        Number of gyroperiods to trace.
    n_step_per_gyro : int, optional
        Number of sampling steps per gyroperiod.
    mass_kg : float, optional
        Ion mass in kg. Use this for custom pickup-ion species.
    charge_c : float, optional
        Ion charge in C.

    Returns
    -------
    dict
        ``T`` in s, ``Pbirth`` in km, ``Vdet`` in km/s, and ``Eev`` in eV.
    """

    particle = _get_species(species, mass_kg=mass_kg, charge_c=charge_c)
    if n_gyro <= 0:
        raise ValueError("n_gyro must be positive.")
    if n_step_per_gyro < 16:
        raise ValueError("n_step_per_gyro must be at least 16.")

    Usw = _as_vector3(Usw, "Usw")
    IMF = _as_vector3(IMF, "IMF")
    Pobs = _as_vector3(Pobs, "Pobs")

    bmag_nt = np.linalg.norm(IMF)
    if bmag_nt <= 0:
        raise ValueError("IMF magnitude must be nonzero.")

    bhat = IMF / bmag_nt
    omega = particle.charge_c * bmag_nt * 1e-9 / particle.mass_kg
    t_gyro = 2.0 * np.pi / omega
    n_step = max(2, int(np.round(n_gyro * n_step_per_gyro)))
    time_s = np.linspace(0.0, n_gyro * t_gyro, n_step + 1)

    u_par = np.dot(Usw, bhat) * bhat
    u_perp = Usw - u_par
    phase = omega * time_s
    cos_phase = np.cos(phase)[:, None]
    sin_phase = np.sin(phase)[:, None]
    u_cross_bhat = np.cross(u_perp, bhat)

    vdet = u_perp * (1.0 - cos_phase) - u_cross_bhat * sin_phase
    dr = (
        u_perp * (time_s[:, None] - sin_phase / omega)
        + u_cross_bhat * ((cos_phase - 1.0) / omega)
    )
    pbirth = Pobs - dr
    eev = 0.5 * particle.mass_kg * (np.linalg.norm(vdet, axis=1) * 1e3) ** 2
    eev = eev / ELEMENTARY_CHARGE_C

    return {"T": time_s, "Pbirth": pbirth, "Vdet": vdet, "Eev": eev}


def _get_species(species: str | None, mass_kg: float | None, charge_c: float) -> PuiSpecies:
    if mass_kg is not None:
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive.")
        if charge_c == 0:
            raise ValueError("charge_c must be nonzero.")
        return PuiSpecies(float(mass_kg), float(charge_c))

    key = str(species).upper()
    if key not in _SPECIES:
        raise ValueError(f"Unsupported species: {species}. Provide mass_kg for a custom species.")
    return _SPECIES[key]


def _as_vector3(value, name: str):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly three components.")
    return arr


__all__ = [
    "pui_birth_curve",
    "pui_birth_curve_one_species",
    "PuiSpecies",
]

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # --- Input parameters with significant By component ---
    usw_val = [-400, 0, 0]  # km/s
    imf_val = [3, 5, 0]  # nT (Bx=3, By=5)
    pobs_val = [4000, 0, 0]  # km (Observation point)

    # --- Calculate H+ and O+ birth curves ---
    # H+: trace more gyros to see the small-scale path
    # O+: trace 1 gyro due to its massive Larmor radius
    curves = pui_birth_curve(Usw=usw_val, IMF=imf_val, Pobs=pobs_val,
                             n_gyro_h=10, n_gyro_o=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot O+ Birth Curve (X-Z Plane) ---
    # Use scatter with 'c' mapped to energy and 's' for point size
    sc1 = ax1.scatter(curves["O"]["Pbirth"][:, 0],
                      curves["O"]["Pbirth"][:, 2],
                      c=curves["O"]["Eev"],
                      s=2, cmap='viridis')
    ax1.set_title('O+ Birth Curve (X-Z)')
    plt.colorbar(sc1, ax=ax1, label='Energy [eV]')

    # --- Plot H+ Birth Curve (X-Z Plane) ---
    sc2 = ax2.scatter(curves["H"]["Pbirth"][:, 0],
                      curves["H"]["Pbirth"][:, 2],
                      c=curves["H"]["Eev"],
                      s=1, cmap='plasma')
    ax2.set_title('H+ Birth Curve (X-Z)')
    plt.colorbar(sc2, ax=ax2, label='Energy [eV]')

    # --- Global styling ---
    for ax in [ax1, ax2]:
        # Mark the observation point (Pobs)
        ax.scatter(pobs_val[0], pobs_val[2], color='black', marker='*', s=150, zorder=5, label='Obs')
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Z [km]')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', 'datalim')  # Keep physical proportions
        ax.legend()

    plt.tight_layout()
    plt.show()
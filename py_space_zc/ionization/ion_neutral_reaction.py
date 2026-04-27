"""
============================================================
Ion Chemistry Reactions for Mars Upper Atmosphere
============================================================

This module defines key ionization, charge exchange, and
dissociative recombination reactions relevant to Mars'
upper atmospheric and ionospheric chemistry.

All input densities must be in [m⁻³].
All velocities must be in [m/s].
All temperatures must be in [K].
All output reaction rates are in [m⁻³/s].

------------------------------------------------------------
Reaction Table
------------------------------------------------------------

# Charge Exchange Reactions
# ---------------------------------------
1.  CO₂   + H⁺  → CO₂⁺ + H             k = 2.00e-15 · V_H⁺   [cm³/s]
2.  O     + H⁺  → O⁺   + H             k = 1.00e-15 · V_H⁺   [cm³/s]
3.  O⁺    + H   → O    + H⁺            k = 6.40e-10          [cm³/s]
4.  CO₂⁺ + O   → O⁺   + CO₂            k = 9.60e-11          [cm³/s]
5.  CO₂⁺ + O   → O₂⁺  + CO             k = 1.64e-10          [cm³/s]
6.  CO₂  + O⁺  → O₂⁺  + CO             k = 1.10e-9 · (800 / Ti)^0.39      [cm³/s]

# Dissociative Recombination
# ---------------------------------------
7.  CO₂⁺ + e⁻  → CO + O                k = 3.10e-7 · (300 / Te)^0.5        [cm³/s]
8.  O₂⁺  + e⁻  → O + O                 k = 7.38e-8 · (1200 / Te)^0.56      [cm³/s]

Note:
    - All rate coefficients above are given in cm³/s.
    - All functions convert cm³/s → m³/s internally.
    - These reactions are based on Schunk & Nagy (2009), and Bougher et al. (2015).

Author: Zhang Chi
Date  : 2025-11-22
"""

import numpy as np


# ============================================================
# 1. CO₂ + H⁺ → CO₂⁺ + H
# ============================================================
def Hp_CO2_to_CO2p_H(nCO2_m3, nHp_m3, Vh_ms):
    """
    Charge exchange: CO₂ + H⁺ → CO₂⁺ + H

    Rate coefficient:
        k = 2e-15 * Vh   [cm³/s]
        where Vh is proton total speed in [m/s]

    Parameters
    ----------
    nCO2_m3 : float or ndarray
        CO₂ neutral density [m⁻³]
    nHp_m3 : float or ndarray
        H⁺ ion density [m⁻³]
    Vh_ms : float or ndarray
        Proton speed [m/s]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 2e-15 * Vh_ms * 1e2   # convert m/s → cm/s
    return k_cm3s * 1e-6 * nCO2_m3 * nHp_m3


# ============================================================
# 2. O + H⁺ → O⁺ + H
# ============================================================
def Hp_O_to_Op_H(nO_m3, nHp_m3, Vh_ms):
    """
    Charge exchange: O + H⁺ → O⁺ + H

    Rate coefficient:
        k = 1e-15 * Vh   [cm³/s]

    Parameters
    ----------
    nO_m3 : float or ndarray
        O neutral density [m⁻³]
    nHp_m3 : float or ndarray
        H⁺ ion density [m⁻³]
    Vh_ms : float or ndarray
        Proton speed [m/s]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 1e-15 * Vh_ms * 1e2
    return k_cm3s * 1e-6 * nO_m3 * nHp_m3


# ============================================================
# 3. O⁺ + H → O + H⁺
# ============================================================
def Op_H_to_Hp_O(nOp_m3, nH_m3):
    """
    Charge exchange: O⁺ + H → O + H⁺

    Rate coefficient:
        k = 6.4e-10   [cm³/s]   (temperature-independent)

    Parameters
    ----------
    nOp_m3 : float or ndarray
        O⁺ ion density [m⁻³]
    nH_m3 : float or ndarray
        Neutral H density [m⁻³]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 6.4e-10
    return k_cm3s * 1e-6 * nOp_m3 * nH_m3


# ============================================================
# 4. CO₂⁺ + O → O⁺ + CO₂
# ============================================================
def CO2p_O_to_Op_CO2(nCO2p_m3, nO_m3):
    """
    Ion-neutral reaction: CO₂⁺ + O → O⁺ + CO₂

    Rate coefficient:
        k = 9.6e-11   [cm³/s]

    Parameters
    ----------
    nCO2p_m3 : float or ndarray
        CO₂⁺ ion density [m⁻³]
    nO_m3 : float or ndarray
        O neutral density [m⁻³]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 9.6e-11
    return k_cm3s * 1e-6 * nCO2p_m3 * nO_m3


# ============================================================
# 5. CO₂⁺ + O → O₂⁺ + CO
# ============================================================
def CO2p_O_to_O2p_CO(nCO2p_m3, nO_m3):
    """
    Ion-neutral reaction: CO₂⁺ + O → O₂⁺ + CO

    Rate coefficient:
        k = 1.64e-10   [cm³/s]

    Parameters
    ----------
    nCO2p_m3 : float or ndarray
        CO₂⁺ ion density [m⁻³]
    nO_m3 : float or ndarray
        O neutral density [m⁻³]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 1.64e-10
    return k_cm3s * 1e-6 * nCO2p_m3 * nO_m3


# ============================================================
# 6. CO₂ + O⁺ → O₂⁺ + CO   (T-dependent)
# ============================================================
def CO2_Op_to_O2p_CP(nCO2_m3, nOp_m3, Ti_K):
    """
    Temperature-dependent ion-neutral reaction:
        CO₂ + O⁺ → O₂⁺ + CO

    Rate coefficient:
        k = 1.1e-9 · (800 / Ti)^0.39  [cm³/s]

    Parameters
    ----------
    nCO2_m3 : float or ndarray
        CO₂ neutral density [m⁻³]
    nOp_m3 : float or ndarray
        O⁺ ion density [m⁻³]
    Ti_K : float or ndarray
        Ion temperature [K]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 1.1e-9 * (800 / Ti_K) ** 0.39
    return k_cm3s * 1e-6 * nCO2_m3 * nOp_m3


# ============================================================
# 7. CO₂⁺ + e⁻ → CO + O  (dissociative recombination)
# ============================================================
def CO2p_e_to_CO_O(nCO2p_m3, ne_m3, Te_K):
    """
    Dissociative recombination: CO₂⁺ + e⁻ → CO + O

    Rate coefficient:
        k = 3.1e-7 * (300 / Te)^0.5   [cm³/s]

    Parameters
    ----------
    nCO2p_m3 : float or ndarray
        CO₂⁺ ion density [m⁻³]
    ne_m3 : float or ndarray
        Electron density [m⁻³]
    Te_K : float or ndarray
        Electron temperature [K]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 3.1e-7 * np.sqrt(300 / Te_K)
    return k_cm3s * 1e-6 * nCO2p_m3 * ne_m3


# ============================================================
# 8. O₂⁺ + e⁻ → O + O  (dissociative recombination)
# ============================================================
def O2p_e_to_O_O(nO2p_m3, ne_m3, Te_K):
    """
    Dissociative recombination: O₂⁺ + e⁻ → O + O

    Rate coefficient:
        k = 7.38e-8 * (1200 / Te)^0.56   [cm³/s]

    Parameters
    ----------
    nO2p_m3 : float or ndarray
        O₂⁺ ion density [m⁻³]
    ne_m3 : float or ndarray
        Electron density [m⁻³]
    Te_K : float or ndarray
        Electron temperature [K]

    Returns
    -------
    rate_m3s : float or ndarray
        Reaction rate [m⁻³/s]
    """
    k_cm3s = 7.38e-8 * (1200 / Te_K) ** 0.56
    return k_cm3s * 1e-6 * nO2p_m3 * ne_m3

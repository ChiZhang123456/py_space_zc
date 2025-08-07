#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Chi Zhang
Email: zhangchi9508@gmail.com
Copyright © 2024–2027
License: MIT
Version: 2.4.10

This module provides a collection of utility functions for plasma physics
data analysis, including VDF manipulation, moment calculation, Monte Carlo
sampling, Maxwellian generation, and coordinate transformations.
"""

# === VDF Utilities ===
from .match_vdf_dims import match_vdf_dims
from .rebin_vdf import rebin_vdf
from .reduce import reduce
from .expand_4d_grid import expand_4d_grid
from .convert_energy_velocity import convert_energy_velocity
from .vxyz_from_polar import vxyz_from_polar
from .get_particle_mass_charge import get_particle_mass_charge

# === Plasma Moments and Distributions ===
from .moments_calculation import moments_calculation
from .plasma_moments import plasma_moments
from .int_sph_dist import int_sph_dist
from .create_pdist_skymap import create_pdist_skymap

# === Maxwellian Distributions ===
from .maxwellian_distribution import _1d as maxwellian_1d
from .maxwellian_distribution import _2d as maxwellian_2d
from .maxwellian_distribution import _3d as maxwellian_3d
from .generate_maxwellian_3d import generate_maxwellian_3d
from .flux_convert import flux_convert
from .rebin_omni import rebin_omni

# === Monte Carlo VDF Sampling ===
from .Monte_Carlo_vdf import _mc_cart_3d as mc_3d
from .Monte_Carlo_vdf import _mc_cart_2d as mc_2d
from .Monte_Carlo_vdf import _mc_pol_1d as mc_1d

__all__ = [
    # VDF Utilities
    "match_vdf_dims",
    "rebin_vdf",
    "reduce",
    "expand_4d_grid",
    "convert_energy_velocity",
    "vxyz_from_polar",
    "get_particle_mass_charge",
    "flux_convert",
    "rebin_omni",

    # Plasma Moments and Distributions
    "moments_calculation",
    "plasma_moments",
    "int_sph_dist",
    "create_pdist_skymap",

    # Maxwellian Distributions
    "maxwellian_1d",
    "maxwellian_2d",
    "maxwellian_3d",
    "generate_maxwellian_3d",

    # Monte Carlo VDF Sampling
    "mc_3d",
    "mc_2d",
    "mc_1d",
]

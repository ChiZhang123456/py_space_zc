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
from .maxwellian_distribution import _bi_2d as bi_maxwellian_2d
from .maxwellian_distribution import _bi_3d as bi_maxwellian_3d
from .generate_maxwellian_3d import generate_maxwellian_3d
from .kappa_distribution import _1d as kappa_1d
from .kappa_distribution import _3d as kappa_3d
from .kappa_distribution import _bi as bi_kappa
from .flux_convert import flux_convert
from .omni_def2psd import omni_def2psd
from .rebin_omni import rebin_omni
from .fit_omni_1d import energy_to_speed as energy_to_speed_1d
from .fit_omni_1d import maxwellian_omni_1d_model
from .fit_omni_1d import kappa_omni_1d_model
from .fit_omni_1d import fit_omni_1d
from .fit_omni_1d import fit_maxwellian_omni_1d
from .fit_omni_1d import fit_kappa_omni_1d
from .fit_2d_bi_psd import bi_maxwellian_2d_psd_model
from .fit_2d_bi_psd import bi_kappa_2d_psd_model
from .fit_2d_bi_psd import fit_2d_bi_psd
from .fit_bi_kappa import energy_to_speed_kms
from .fit_bi_kappa import energy_pitchangle_to_vpara_vperp
from .fit_bi_kappa import bi_kappa_energy_pa_model
from .fit_bi_kappa import fit_bi_kappa_energy_pa
from .fit_bi_kappa import fit_bi_kappa
from .fit_bi_kappa import plot_bi_kappa_fit
from .fit_3d_psd import maxwellian_3d_psd_model
from .fit_3d_psd import kappa_3d_psd_model
from .fit_3d_psd import fit_3d_psd

# === Monte Carlo VDF Sampling ===
from .Monte_Carlo_vdf import _mc_cart_3d as mc_3d
from .Monte_Carlo_vdf import _mc_cart_2d as mc_2d
from .Monte_Carlo_vdf import _mc_pol_1d as mc_1d


# === Calculate the pitch angle distribution ===
from .pitchangle_dis import pitchangle_dis
from .pitchangle_merge_energy import pitchangle_merge_energy
from .pitchangle_merge_pa import pitchangle_merge_pa
from .pitchangle_dis_3d import pitchangle_dis_3d
from .gyrophase_pitchangle_dis import gyrophase_pitchangle_dis
from .gyrophase_pitchangle_dis import gyrophase_pitchangle_dis_3d
from .monte_carlo_pad import par_perp_reduced_dis as par_perp_reduced_dis

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
    "omni_def2psd",
    "rebin_omni",
    "energy_to_speed_1d",

    # Plasma Moments and Distributions
    "moments_calculation",
    "plasma_moments",
    "int_sph_dist",
    "create_pdist_skymap",

    # Maxwellian Distributions
    "maxwellian_1d",
    "maxwellian_2d",
    "maxwellian_3d",
    "bi_maxwellian_2d",
    "bi_maxwellian_3d",
    "generate_maxwellian_3d",
    "kappa_1d",
    "kappa_3d",
    "bi_kappa",
    "maxwellian_omni_1d_model",
    "kappa_omni_1d_model",
    "fit_omni_1d",
    "fit_maxwellian_omni_1d",
    "fit_kappa_omni_1d",
    "bi_maxwellian_2d_psd_model",
    "bi_kappa_2d_psd_model",
    "fit_2d_bi_psd",
    "energy_to_speed_kms",
    "energy_pitchangle_to_vpara_vperp",
    "bi_kappa_energy_pa_model",
    "fit_bi_kappa_energy_pa",
    "fit_bi_kappa",
    "plot_bi_kappa_fit",
    "maxwellian_3d_psd_model",
    "kappa_3d_psd_model",
    "fit_3d_psd",

    # Monte Carlo VDF Sampling
    "mc_3d",
    "mc_2d",
    "mc_1d",


    # pitch angle distribution
    "pitchangle_dis",
    "pitchangle_merge_energy",
    "pitchangle_merge_pa",
    "pitchangle_dis_3d",
    "gyrophase_pitchangle_dis",
    "gyrophase_pitchangle_dis_3d",
    "par_perp_reduced_dis",
]

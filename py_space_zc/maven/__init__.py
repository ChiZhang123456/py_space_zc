#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAVEN-related tools for SPICE data, plasma & magnetic field analysis,
position/geometry utilities, plotting helpers, and model integration.

Author: Chi Zhang
Email: zhangchi9508@gmail.com
Copyright: 2024–2027
License: MIT
"""

# -----------------------------------------------------------------------------
# Base utilities
# -----------------------------------------------------------------------------
from .db_init import db_init
from .get_base_path import get_base_path
from .load_maven_spice import load_maven_spice
from .spice_init import spice_init
from . import static, swia, swea  # ensure submodules are importable at package level
from .mars_neutral_density import mars_neutral_density
# -----------------------------------------------------------------------------
# Data retrieval & transforms
# -----------------------------------------------------------------------------
from .get_data import get_data, load_data
from .coords_convert import coords_convert
from .get_lang_19_map import get_lang_19_map
from .show_overview import show_overview
from .show_maven_data import show_maven_data
# -----------------------------------------------------------------------------
# Position & geometry
# -----------------------------------------------------------------------------
from .get_pos import (
    get_pos_spice,
    get_subsolar,
    get_lon_lat,
    get_sza_alt,
    get_midnight,
)

# -----------------------------------------------------------------------------
# Coordinate projection
# -----------------------------------------------------------------------------
from .lonlat_pc import lonlat2pc, pc2lonlat

# -----------------------------------------------------------------------------
# Crustal field models
# -----------------------------------------------------------------------------
from .crustal_fields import (
    cf_model,
    cf_model_pc,
    cf_model_mso,
)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
from .plot_data import (
    plot_B,
    plot_B_high,
    plot_B_mse,
    plot_swia_omni,
    plot_swia_pad,
    plot_swia_reduced_2d,
    plot_swia_vpar_perp,
    plot_swea_omni,
    plot_swea_pad,
    plot_sta_c6,
    plot_sta_c8,
    plot_d1_reduced_2d,
    plot_d1_reduced_1d,
    plot_crustal_field_map,
    plot_sta_dens,
    plot_d1_flux,
)
from .show_path_xr import show_path_xr

# -----------------------------------------------------------------------------
# Model boundaries & diagnostics
# -----------------------------------------------------------------------------
from .bs_mpb import bs_mpb
from .BS_MPB_3d_mars import BS_MPB_3d_mars
from .mpb_tangent_direction import mpb_tangent_direction
from .mpb_normal import mpb_normal
from .In_MPB import In_MPB
from .get_vsc_mso import get_vsc_mso

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # base
    "db_init",
    "get_base_path",
    "load_maven_spice",
    "spice_init",
    "static",
    "swia",
    "swea",
    "mars_neutral_density",
    # data & transforms
    "get_data",
    "load_data",
    "coords_convert",
    "get_lang_19_map",
    "show_overview",
    'show_maven_data',

    # position & geometry
    "get_pos_spice",
    "get_subsolar",
    "get_lon_lat",
    "get_sza_alt",
    "get_midnight",

    # coordinate projection
    "lonlat2pc",
    "pc2lonlat",

    # crustal field models
    "cf_model",
    "cf_model_pc",
    "cf_model_mso",

    # plotting
    "plot_B",
    "plot_B_high",
    "plot_B_mse",
    "plot_swia_omni",
    "plot_swia_pad",
    "plot_swia_reduced_2d",
    "plot_swia_vpar_perp",
    "plot_swea_omni",
    "plot_swea_pad",
    "plot_sta_c6",
    "plot_sta_c8",
    "plot_d1_reduced_2d",
    "plot_d1_reduced_1d",
    "plot_crustal_field_map",
    "plot_sta_dens",
    "plot_d1_flux",
    "show_path_xr",

    # boundaries & diagnostics
    "bs_mpb",
    "BS_MPB_3d_mars",
    "get_vsc_mso",
    "In_MPB",
    "mpb_tangent_direction",
    "mpb_normal",

]

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
__author__ = "Chi Zhang"
__email__ = "zhangchi9508@gmail.com"
__license__ = "MIT"


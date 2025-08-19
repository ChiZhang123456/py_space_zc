#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAVEN-related tools for SPICE data, plasma and magnetic field analysis,
position conversion, plotting, and model integration.

Author: Chi Zhang
Email: zhangchi9508@gmail.com
Copyright 2024–2027
License: MIT
Version: 2.4.10
"""

# === Base utilities ===
from .db_init import db_init
from .download_data import download_data
from .get_base_path import get_base_path
from .load_maven_spice import load_maven_spice
from . import static
# === Data retrieval and transformation ===
from .get_data import get_data
from .coords_convert import coords_convert
from .get_lang_19_map import get_lang_19_map

from .spice_init import spice_init

# === Position and geometry ===
from .get_pos import (
    get_pos_spice,
    get_subsolar,
    get_lon_lat,
    get_sza_alt,
    get_midnight,
)

# === Coordinate projection ===
from .lonlat_pc import lonlat2pc, pc2lonlat

# === Crustal field models ===
from .crustal_fields import (
    cf_model,
    cf_model_pc,
    cf_model_mso,
)

# === Plotting ===
from .plot_data import (
    plot_B,
    plot_B_high,
    plot_B_mse,
    plot_swia_omni,
    plot_swea_omni,
    plot_swea_epad,
    plot_sta_c6,
    plot_d1_reduced_2d,
    plot_crustal_field_map,
)
from .show_path_xr import show_path_xr

# === Model boundaries ===
from .bs_mpb import bs_mpb
from .BS_MPB_3d_mars import BS_MPB_3d_mars
from .get_vsc_mso import get_vsc_mso

__all__ = [
    # base
    "db_init",
    "download_data",
    "get_base_path",
    "load_maven_spice",

    # data & transforms
    "get_data",
    "coords_convert",
    "get_lang_19_map",
    "spice_init",

# Instrument
    "static",
    "swia",

    # position
    "get_pos_spice",
    "get_subsolar",
    "get_lon_lat",
    "get_sza_alt",
    "get_midnight",

    # coordinates
    "lonlat2pc",
    "pc2lonlat",

    # crustal field models
    "cf_model",
    "cf_model_pc",
    "cf_model_mso",

    # plots
    "plot_B",
    "plot_B_high",
    "plot_B_mse",
    "plot_swia_omni",
    "plot_swea_omni",
    "plot_swea_epad",
    "show_path_xr",
    "plot_sta_c6",
    "plot_d1_reduced_2d",
    "plot_crustal_field_map",

    # boundary model
    "bs_mpb",
    "BS_MPB_3d_mars",
    "get_vsc_mso",
]

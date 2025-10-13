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
from .get_base_path import get_base_path
# === Data retrieval and transformation ===
from .get_data import get_data
from .extract_data_minpa import extract_data_minpa
from .minpa_omni import minpa_omni
from .plot_data import plot_B
from .plot_data import plot_minpa_mod1_omni
from .plot_maven_tianwen_yz import plot_maven_tianwen_yz
from .plot_maven_tianwen_xr import plot_maven_tianwen_xr
from .plot_maven_tianwen_clock_angle import plot_maven_tianwen_clock_angle
from .show_maven_tianwen_data import show_maven_tianwen_data
from .plot_maven_tianwen_dca import plot_maven_tianwen_dca
from .plot_maven_tianwen_yz_mse import plot_maven_tianwen_yz_mse


__all__ = [
    # base
    "db_init",
    "get_base_path",

    # data & transforms
    "get_data",
    "extract_data_minpa",
    "minpa_omni"
    "plot_B",
    "plot_minpa_mod1_omni",
    #
    "plot_maven_tianwen_xr",
    "plot_maven_tianwen_yz",
    "plot_maven_tianwen_clock_angle",
    "show_maven_tianwen_data",
    "plot_maven_tianwen_dca",
]

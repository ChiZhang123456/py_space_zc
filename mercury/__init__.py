#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mercury Space Environment Utilities

This package provides tools for plotting empirical boundary models 
(bow shock and magnetopause) around Mercury, based on published fits 
from spacecraft observations (e.g., Zhong et al. 2015, Winslow et al. 2013).

Includes:
---------
- Bow shock and magnetopause model plotting functions
- Ready-to-use compound visualizations (e.g., `bs_mpb`)

Author: Chi Zhang
Email: zhangchi9508@gmail.com
Copyright © 2024–2027
License: MIT
Version: 2.4.10
"""

# === Boundary model utilities ===
from .bs_mpb import bs_mpb, add_bs, add_mp_winslow, add_mp_zhong

__all__ = [
    "bs_mpb",           # Combined bow shock + MP + disk visualization
    "add_bs",           # Winslow et al. (2013) bow shock
    "add_mp_winslow",   # Winslow et al. (2013) magnetopause
    "add_mp_zhong",     # Zhong et al. (2015) magnetopause
]

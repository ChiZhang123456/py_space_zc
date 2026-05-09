#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Chi Zhang
Email: zhangchi9508@gmail.com
Copyright © 2024–2027
License: MIT
Version: 2.4.10
"""

from .mva import mva
from .mva_gui import mva_gui
from .SVD_B import SVD_B
from .wavelet import wavelet
from .shock_gui import shock_gui
from .walen_test import walen_test
from .welch_psd import psd_welch
from .welch_psd import psd_welch_sliding
from .fac import fac
from .autocorr import autocorr
from .icw_detection import detect_pickup_icw, detect_pcw_svd, detect_pcw_svd_5min
from .pui_birth_curve import (
    pui_birth_curve,
    pui_birth_curve_one_species,
)
from .beta_anisotropy_instability import (
    plot_beta_anisotropy_instability,
    check_beta_anisotropy_instability,
)
from .current_sheet_fit import fit_current_sheet

__all__ = [
    "mva_gui",
    "mva",
    "SVD_B",
    "wavelet",
    "shock_gui",
    "walen_test",
    "psd_welch",
    "psd_welch_sliding",
    "fac",
    "autocorr",
    "detect_pickup_icw",
    "detect_pcw_svd",
    "detect_pcw_svd_5min",
    "pui_birth_curve",
    "pui_birth_curve_one_species",
    "plot_beta_anisotropy_instability",
    "check_beta_anisotropy_instability",
    "fit_current_sheet",
]

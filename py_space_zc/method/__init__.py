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
from .hilbert_envelope import hilbert_envelope
from .autocorr import autocorr
from .icw_detection import detect_pcw_svd_psd_criteria, evaluate_pcw_svd_psd_criteria_window
from .romeo_pcw_detection import detect_pcw_welch_mva, plot_pcw_welch_mva
from .pui_birth_curve import (
    pui_birth_curve,
    pui_birth_curve_one_species,
)
from .beta_anisotropy_instability import (
    plot_beta_anisotropy_instability,
    check_beta_anisotropy_instability,
)
from .current_sheet_fit import fit_current_sheet
from .fermi_betatron import (
    def_to_psd_shape,
    fermi_betatron_error_map,
    fermi_betatron_map,
    fit_fermi_betatron,
)

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
    "hilbert_envelope",
    "autocorr",
    "detect_pcw_svd_psd_criteria",
    "evaluate_pcw_svd_psd_criteria_window",
    "detect_pcw_welch_mva",
    "plot_pcw_welch_mva",
    "pui_birth_curve",
    "pui_birth_curve_one_species",
    "plot_beta_anisotropy_instability",
    "check_beta_anisotropy_instability",
    "fit_current_sheet",
    "def_to_psd_shape",
    "fermi_betatron_error_map",
    "fermi_betatron_map",
    "fit_fermi_betatron",
]

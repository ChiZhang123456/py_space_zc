# -*- coding: utf-8 -*-
"""
Main package initializer for py_space_zc.

This package provides a collection of utilities for space plasma
data analysis, including:
- Time handling
- Coordinate transformations
- Vector and tensor operations
- Data structures for time series (ts_*)
- MAVEN, EMM, Tianwen-1, mission tools
- Space Physics methods (e.g., MVA)
- Visualization helpers

Author: Chi Zhang
Email: zhangchi9508@gmail.com
Created: 2024-07-24
License: MIT
Version: 2.4.2
Status: Prototype
"""

# -----------------------------------------------------------------------------
# Core imports
# -----------------------------------------------------------------------------
from .cdf_zc import get_cdf_var_info as cdf_var_info, get_cdf_var
from .time_linspace import time_linspace
from .year_month_day import year_month_day
from .adjust_time import adjust_time
from .dir_file import dir_file
from .trig_deg import sind, cosd, tand, asind, acosd, atand, cotd, atan2d
from .irf_time import irf_time
from .pad_split_energy import pad_split_energy
from .nanmean_longitude import nanmean_longitude
from .get_filelist import get_filelist
from .tint_data import tint_data
from .vec_contran_coord import sph2cart_vec, pol2cart_vec, cart2pol_vec, cart2sph_vec
from .coordinate_transform import cart2sph, sph2cart, cart2pol, pol2cart
from .norm_normalize import norm, normalize
from .movemean import movemean
from .add_sphere import add_sphere
from .ang import ang
from .cone_clock_angle import cone_clock_angle
from .read_time_from_file import read_time_from_file
from .generate_line import generate_line
from .generate_plane import generate_plane
from .dot import dot
from .cross import cross
from .lmn import lmn
from .resample_time import resample_time
from .ts_spectr import ts_spectr
from .ts_vec_xyz import ts_vec_xyz
from .ts_scalar import ts_scalar
from .ts_append import ts_append
from .ts_skymap import ts_skymap
from .ts_tensor_xyz import ts_tensor_xyz
from .e_vxb import e_vxb
from .smooth import smooth
from .loadmat import loadmat
from .delta_angle import delta_angle
from .KH_condition import KH_condition
from .time_eval import time_eval
from .background_B import background_B
from .plot_Bwave_svd import plot_Bwave_svd
from .pressure import pressure
from .gyro_information import gyro_information
from .rotate_tensor import rotate_tensor
from .xyz_2_lonlat import xyz_2_lonlat
from .lonlat_2_xyz import lonlat_2_xyz
# -----------------------------------------------------------------------------
# Submodules
# -----------------------------------------------------------------------------
from . import maven, emm, vdf, mercury, vlasitor, ionization, tianwen_1, plot, method, jupiter, sputtering, vex

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
__author__ = "Chi Zhang"
__email__ = "zhangchi9508@gmail.com"
__copyright__ = "Copyright 2024-2027"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # File and data loaders
    "loadmat", "cdf_var_info", "get_cdf_var", "get_filelist",
    "read_time_from_file", "dir_file",

    # Time utilities
    "time_linspace", "year_month_day", "adjust_time", "irf_time",
    "tint_data", "resample_time","time_eval",

    # Math / statistics
    "sind", "cosd", "tand", "asind", "acosd", "atand", "atan2d", "cotd",
    "ang", "delta_angle", "nanmean_longitude", "smooth","movemean",'background_B',

    # Vector and coordinate transforms
    "dot", "cross", "lmn",
    "norm", "normalize",
    "sph2cart_vec", "pol2cart_vec", "cart2pol_vec", "cart2sph_vec",
    "cart2sph", "sph2cart", "cart2pol", "pol2cart",
    "cone_clock_angle", "e_vxb",

    # Geometric utilities
    "add_sphere", "generate_line", "generate_plane",

    # Time series data structures
    "ts_scalar", "ts_vec_xyz", "ts_spectr", "ts_skymap",
    "ts_tensor_xyz", "ts_append",

    # Physics methods
    "pad_split_energy", "KH_condition", "plot_Bwave_svd",

    'pressure', 'gyro_information', 'rotate_tensor', "xyz_2_lonlat","lonlat_2_xyz",

    # Submodules
    "maven", "emm", "vdf", "mercury", "vlasitor", "ionization",
    "tianwen_1", "plot", "method", "jupiter", "sputtering", "vex",
]

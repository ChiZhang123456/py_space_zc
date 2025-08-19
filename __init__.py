# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:41:31 2024

@author: Chi Zhang
"""

from .cdf_zc import get_cdf_var_info as cdf_var_info
from .cdf_zc import get_cdf_var as get_cdf_var
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
from .vec_contran_coord import (
    sph2cart_vec, pol2cart_vec, cart2pol_vec, cart2sph_vec
)
from .coordinate_transform import cart2sph, sph2cart, cart2pol, pol2cart
from .norm_normalize import norm, normalize
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
from . import maven, emm, vdf, mercury, vlasitor, ionization, tianwen_1, plot

__author__ = "Chi Zhang"
__email__ = "zhangchi9508@gmail.com"
__copyright__ = "Copyright 2024-2027"
__license__ = "MIT"
__version__ = "2.4.2"
__status__ = "Prototype"

__all__ = [
    "loadmat",
    "e_vxb",
    "smooth",
    "ts_append",
    "ts_scalar",
    "ts_vec_xyz",
    "ts_spectr",
    "ts_skymap",
    "ts_tensor_xyz",
    "resample_time",
    "cdf_var_info",
    "get_cdf_var",
    "get_filelist",
    "tint_data",
    "time_linspace",
    "year_month_day",
    "adjust_time",
    "irf_time",
    "pad_split_energy",
    "nanmean_longitude",
    "cart2sph_vec",
    "cart2pol_vec",
    "sph2cart_vec",
    "pol2cart_vec",
    "cart2sph,"
    "sph2cart,"
    "cart2pol,"
    "pol2cart,"
    "dir_file",
    "dot",
    "cross",
    "lmn",
    "cosd",
    "sind",
    "tand",
    "asind",
    "acosd",
    "atand",
    "atan2d",
    "cotd",
    "ang",
    "cone_clock_angle",
    "norm",
    "normalize",
    "add_sphere",
    "read_time_from_file",
    "generate_line",
    "generate_plane",
    "maven",
    "emm",
    "vdf",
    "mercury",
    "vlasitor",
    "ionization",
    "tianwen_1",
    "plot",

]

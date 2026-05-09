# 3rd party imports
import numpy as np
import xarray as xr

# Local imports
from .resample import resample
from .ts_vec_xyz import ts_vec_xyz
from .e_vxb import e_vxb

def v_exb(v, B):
    """Compute the ExB drift velocity from electric drift and magnetic field.

    Author: Chi Zhang

    This placeholder is kept for API compatibility. Use `e_vxb` when computing
    the electric field from velocity and magnetic field.
    """
    raise NotImplementedError("v_exb is not implemented yet.")

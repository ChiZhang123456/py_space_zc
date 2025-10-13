import numpy as np
from py_space_zc import ts_scalar
from pyrfu.pyrf import resample
import xarray as xr

def delta_angle(ang1, ang2):
    """
    Compute wrapped angular difference (ang1 - ang2) in degrees, mapped to [-180, 180].

    Behavior:
    - If `ang1` is an xarray.DataArray:
        * If `ang2` is an xarray.DataArray, it is resampled to `ang1`'s time grid.
        * If `ang2` is a scalar or a NumPy array:
            - scalar: broadcast to `ang1.shape`
            - array: must match `ang1.shape`
        * Returns an xarray.DataArray via ts_scalar(ang1.time, diff).
    - Otherwise (NumPy/scalars): returns a NumPy array.

    Parameters
    ----------
    ang1, ang2 : xarray.DataArray or array-like
        Angles in degrees.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Wrapped difference in [-180, 180] degrees.
    """

    def _wrap_deg(diff):
        # Map to [-180, 180]; NaNs are preserved
        return ((diff + 180.0) % 360.0) - 180.0

    # Case 1: xarray path (preferred for time series)
    if isinstance(ang1, xr.DataArray):
        a1 = np.asarray(ang1.data)

        if isinstance(ang2, xr.DataArray):
            # Resample ang2 onto ang1's time grid (correct argument order)
            ang2_rs = resample(ang2, ang1)
            a2 = np.asarray(ang2_rs.data)
        else:
            # scalar or ndarray
            a2 = np.asarray(ang2)
            if a2.ndim == 0:
                a2 = np.full_like(a1, float(a2))
            elif a2.shape != a1.shape:
                raise ValueError(
                    "When ang1 is a DataArray, ang2 must be a scalar, "
                    "have the same shape, or be a DataArray to be resampled."
                )

        diff = np.abs(_wrap_deg(a1 - a2))
        return ts_scalar(ang1.time.data, diff)

    # Case 2: pure NumPy / scalar inputs
    a1 = np.asarray(ang1)
    a2 = np.asarray(ang2)
    diff = np.abs(_wrap_deg(a1 - a2))

    return diff

if __name__ == "__main__":
    res = delta_angle(200.0, 20.0)
    print(res)
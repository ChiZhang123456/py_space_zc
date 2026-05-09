import numpy as np
from pyrfu.pyrf import ts_vec_xyz, convert_fac


def fac(B, r_xyz=None):
    """
    Convert vector time series B to FAC coordinates using mean B as background.
    The returned FAC system is fully consistent with pyrfu.convert_fac.

    FAC definition follows pyrfu:
        z_fac (fac_z) : parallel direction (along background magnetic field B0)
        y_fac (fac_y) : perpendicular direction = B0 × r_xyz
        x_fac (fac_x) : perpendicular direction = y_fac × B0

    IMPORTANT:
        The output B_fac has the following component order:

            B_fac[:, 0] → B_perp1  (along fac_x)  ⟂ B0
            B_fac[:, 1] → B_perp2  (along fac_y)  ⟂ B0
            B_fac[:, 2] → B_para   (along fac_z)  ∥ B0

        i.e.,
            B_fac = [B_perp1, B_perp2, B_parallel]

    Parameters
    ----------
    B : xarray.DataArray
        Vector time series, shape = (time, 3).
    r_xyz : array-like, optional
        Reference vector used to define perpendicular directions.
        If None, defaults to [1, 0, 0], consistent with pyrfu.convert_fac.

    Returns
    -------
    B_fac : xarray.DataArray
        Magnetic field in FAC coordinates, shape = (time, 3):
        [B_perp1, B_perp2, B_parallel].

    fac_x : xarray.DataArray
        FAC x unit vector (perp1 direction).

    fac_y : xarray.DataArray
        FAC y unit vector (perp2 direction).

    fac_z : xarray.DataArray
        FAC z unit vector (parallel to B0).

    b_bgd : xarray.DataArray
        Background magnetic field used to define FAC.
    """

    if r_xyz is None:
        r_xyz = np.array([1.0, 0.0, 0.0])
    else:
        r_xyz = np.asarray(r_xyz, dtype=float)

    # mean background magnetic field
    b0_vec = np.nanmean(B.data, axis=0)

    if np.any(~np.isfinite(b0_vec)):
        raise ValueError("Mean background magnetic field contains NaN or Inf.")

    if np.linalg.norm(b0_vec) == 0:
        raise ValueError("Mean background magnetic field has zero magnitude.")

    # make background field time series, same length as B
    b_bgd_data = np.tile(b0_vec, (len(B.time), 1))

    b_bgd = ts_vec_xyz(
        B.time.data,
        b_bgd_data,
        attrs=B.attrs
    )

    # same as pyrfu.convert_fac
    B_fac = convert_fac(B, b_bgd, r_xyz)

    # construct FAC basis exactly following convert_fac.py
    b_hat = b_bgd_data / np.linalg.norm(
        b_bgd_data, axis=1, keepdims=True
    )

    r_xyz_data = np.tile(r_xyz, (len(B.time), 1))

    fac_y_data = np.cross(b_hat, r_xyz_data, axis=1)
    fac_y_norm = np.linalg.norm(fac_y_data, axis=1, keepdims=True)

    if np.any(fac_y_norm == 0):
        raise ValueError(
            "FAC y is undefined because background B is parallel to r_xyz. "
            "Try another r_xyz, e.g. [0, 1, 0]."
        )

    fac_y_data = fac_y_data / fac_y_norm

    fac_x_data = np.cross(fac_y_data, b_bgd_data, axis=1)
    fac_x_data = fac_x_data / np.linalg.norm(
        fac_x_data, axis=1, keepdims=True
    )

    fac_z_data = b_hat

    fac_x = ts_vec_xyz(B.time.data, fac_x_data)
    fac_y = ts_vec_xyz(B.time.data, fac_y_data)
    fac_z = ts_vec_xyz(B.time.data, fac_z_data)

    return B_fac, fac_x, fac_y, fac_z, b_bgd
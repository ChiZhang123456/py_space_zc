import numpy as np
from typing import Union, Iterable
import copy
from py_space_zc import normalize, cross, ts_tensor_xyz, ts_vec_xyz, maven, vdf, dot, time_eval
from pyrfu import pyrf
import spiceypy as sp

def _tile_to_ts_vec(time, vec: np.ndarray, name: str = "axis"):
    """
    Ensure `vec` is a (nt, 3) array and wrap as ts_vec_xyz aligned with `time`.
    Accepts shapes (3,), (1,3), or (nt,3). Raises if shape incompatible.
    """
    vec = np.asarray(vec)

    if vec.ndim == 1 and vec.shape[0] == 3:
        arr = np.tile(vec[np.newaxis, :], (len(time), 1))
    elif vec.ndim == 2 and vec.shape == (1, 3):
        arr = np.tile(vec, (len(time), 1))
    elif vec.ndim == 2 and vec.shape[1] == 3 and vec.shape[0] == len(time):
        arr = vec
    else:
        raise ValueError(
            f"{name} must be shape (3,), (1,3) or (nt,3) with nt=len(time); got {vec.shape}"
        )

    return ts_vec_xyz(time, arr)


def _ensure_orthonormal_triad(e1: np.ndarray, e2: np.ndarray, tol: float = 1e-6):
    """
    Given two (nt,3) arrays e1,e2 (not necessarily unit or orthogonal),
    return a right-handed orthonormal triad (e1h,e2h,e3h).
    If e1 and e2 are nearly parallel, raises ValueError.
    """
    e1n = normalize(e1)
    # Remove any component of e2 along e1, then normalize
    e2p = e2 - (np.sum(e2 * e1n, axis=1, keepdims=True)) * e1n
    # parallel check
    if np.any(np.linalg.norm(e2p, axis=1) < tol):
        raise ValueError("mso_axis1 and mso_axis2 are parallel or nearly parallel.")

    e2n = normalize(e2p)
    e3n = normalize(cross(e1n, e2n))  # right-handed

    return e1n, e2n, e3n


def reduced_swia_2d(
    swia_3d,
    mso_axis1: Union[Iterable[float], np.ndarray],
    mso_axis2: Union[Iterable[float], np.ndarray],
    vg_2d: Union[np.ndarray, Iterable[float], None] = None,
):
    """
    Reduce a 3D SWIA velocity distribution onto a 2D plane defined by two MSO-space axes.

    Parameters
    ----------
    swia_3d : xarray.Dataset
        MAVEN SWIA 3D ion distribution (DEF, energy, angles, etc.)
    mso_axis1, mso_axis2 : array-like
        Two linearly independent vectors in MSO coordinates. Shapes: (3,), (1,3), or (nt,3)
    vg_2d : array-like, optional
        Velocity grid (1D) in km/s. If None, defaults to [-1000, 1000] km/s with 500 bins.

    Returns
    -------
    f2D : xarray.DataArray
        2D reduced distribution in instrument frame, over (v1, v2) plane.
    """
    # --- SPICE init
    sp.kclear()
    maven.load_maven_spice()

    # --- Prepare basis vectors in MSO, tile to match time axis
    t_arr = swia_3d.time.data
    ax1_ts = _tile_to_ts_vec(t_arr, np.asarray(mso_axis1), "mso_axis1")  # (nt,3)
    ax2_ts = _tile_to_ts_vec(t_arr, np.asarray(mso_axis2), "mso_axis2")  # (nt,3)

    # --- Transform axes to SWIA instrument frame (STA)
    ax1_swia = maven.coords_convert(ax1_ts, 'mso2swia')  # (nt,3)
    ax2_swia = maven.coords_convert(ax2_ts, 'mso2swia')  # (nt,3)

    # --- Construct orthonormal triad [e1 e2 e3] in SWIA frame
    e1, e2, e3 = _ensure_orthonormal_triad(ax1_swia, ax2_swia)
    basis_tensor = np.stack([e1, e2, e3], axis=2)  # (nt, 3, 3)
    xyz = ts_tensor_xyz(t_arr, basis_tensor)

    # --- Velocity grid: convert to m/s
    if vg_2d is None:
        vg_2d = np.linspace(-1000.0, 1000.0, 500) * 1e3
    else:
        vg_2d = np.asarray(vg_2d, dtype=float) * 1e3

    # --- Convert DEF to PSD
    psd = vdf.flux_convert(swia_3d, "def2psd")

    # --- Perform 2D reduction using Monte Carlo integration
    f2D = vdf.reduce(
        psd,
        xyz=xyz,
        dim="2d",
        base="cart",
        vg=vg_2d,
        sc_pot=None,
        n_mc=1000
    )

    # --- Clear SPICE after use
    sp.kclear()

    return f2D

if __name__ == "__main__":
    maven.load_maven_spice()
    tint = ["2018-10-18T20:10:00", "2018-10-18T21:10:00"]
    swia_3d = maven.get_data(tint, 'swia_3d')
    xmso = np.array((1.0,0.0,0.0))
    ymso = np.array((0.0,1.0,0.0))
    t1 = np.datetime64('2018-10-18T20:30:00')
    f2d = reduced_swia_2d(swia_3d, xmso, ymso)
    sp.kclear()
    f2d_t1 = time_eval(f2d, t1)
    from py_space_zc import plot
    import matplotlib.pyplot as plt
    f2d_t1.data = np.where(f2d_t1.data > 1e-7, f2d_t1.data, np.nan)
    ax, cax = plot.plot_spectr(None, f2d_t1, cmap = 'Spectral_r', cscale='log', clim = [1e-7, 1e-4])
    ax.set_xlim(-500.0, 500.0)
    ax.set_ylim(-500.0, 500.0)
    ax.set_aspect('equal')
    plt.show()
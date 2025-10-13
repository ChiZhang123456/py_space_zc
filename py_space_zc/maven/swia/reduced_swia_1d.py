import numpy as np
from typing import Union, Iterable
import copy
import spiceypy as sp
import matplotlib.pyplot as plt

from py_space_zc import (
    normalize,
    cross,
    ts_tensor_xyz,
    ts_vec_xyz,
    maven,
    vdf,
    dot,
    time_eval,
    plot,
)
from pyrfu import pyrf


def _tile_to_ts_vec(time, vec: np.ndarray, name: str = "axis"):
    """
    Ensure `vec` is expanded to match the length of `time`
    and returned as a ts_vec_xyz object.

    Parameters
    ----------
    time : array-like
        Time array (length nt).
    vec : np.ndarray
        Vector input. Acceptable shapes:
        - (3,)     : single vector
        - (1, 3)   : single row vector
        - (nt, 3)  : time-dependent vector
    name : str, optional
        Label for error reporting.

    Returns
    -------
    ts_vec_xyz
        Vector time series aligned with `time`.

    Raises
    ------
    ValueError
        If input shape is incompatible.
    """
    vec = np.asarray(vec)

    if vec.ndim == 1 and vec.shape[0] == 3:
        # Broadcast single vector across all times
        arr = np.tile(vec[np.newaxis, :], (len(time), 1))
    elif vec.ndim == 2 and vec.shape == (1, 3):
        # Broadcast row vector across all times
        arr = np.tile(vec, (len(time), 1))
    elif vec.ndim == 2 and vec.shape[1] == 3 and vec.shape[0] == len(time):
        # Already time-dependent
        arr = vec
    else:
        raise ValueError(
            f"{name} must be shape (3,), (1,3), or (nt,3) with nt=len(time); got {vec.shape}"
        )

    return ts_vec_xyz(time, arr)


def _ensure_orthonormal_triad(e1: np.ndarray, e2: np.ndarray, tol: float = 1e-6):
    """
    Construct a right-handed orthonormal basis given two input vectors.

    Parameters
    ----------
    e1, e2 : np.ndarray
        Input arrays of shape (nt, 3). They do not need to be normalized
        or orthogonal initially.
    tol : float, optional
        Tolerance for detecting nearly parallel vectors.

    Returns
    -------
    (e1n, e2n, e3n) : tuple of np.ndarray
        Right-handed orthonormal triad.

    Raises
    ------
    ValueError
        If e1 and e2 are nearly parallel.
    """
    # Normalize first axis
    e1n = normalize(e1)

    # Remove projection of e2 along e1
    e2p = e2 - (np.sum(e2 * e1n, axis=1, keepdims=True)) * e1n

    # Check for degeneracy
    if np.any(np.linalg.norm(e2p, axis=1) < tol):
        raise ValueError("Input axes are parallel or nearly parallel.")

    # Normalize second axis
    e2n = normalize(e2p)

    # Third axis via cross product (ensures right-handed basis)
    e3n = normalize(cross(e1n, e2n))

    return e1n, e2n, e3n


def reduced_swia_1d(
    swia_3d,
    mso_axis1: Union[Iterable[float], np.ndarray],
    vg_1d: Union[np.ndarray, Iterable[float], None] = None,
):
    """
    Reduce a MAVEN SWIA 3D ion velocity distribution to a 1D representation
    along a plane defined by two MSO-space axes.

    Parameters
    ----------
    swia_3d : xarray.Dataset
        MAVEN SWIA 3D ion distribution (DEF, energy, angles, etc.).
    mso_axis1 : array-like
        Primary axis in MSO coordinates. Shapes allowed:
        (3,), (1,3), or (nt,3).
    vg_1d : array-like, optional
        1D velocity grid in km/s. Defaults to [-1000, 1000] km/s (500 bins).

    Returns
    -------
    f1D : xarray.DataArray
        Reduced 1D distribution in instrument frame.
    """
    # --- SPICE setup
    sp.kclear()
    maven.load_maven_spice()

    # --- Prepare time axis
    t_arr = swia_3d.time.data

    # --- Tile axis inputs to time dimension
    ax1_ts = _tile_to_ts_vec(t_arr, np.asarray(mso_axis1), "mso_axis1")
    mso_axis2 = normalize(np.ones_like(mso_axis1))  # Placeholder orthogonal vector
    ax2_ts = _tile_to_ts_vec(t_arr, np.asarray(mso_axis2), "mso_axis2")

    # --- Transform axes into SWIA instrument frame
    ax1_swia = maven.coords_convert(ax1_ts, "mso2swia")
    ax2_swia = maven.coords_convert(ax2_ts, "mso2swia")

    # --- Build orthonormal triad
    e1, e2, e3 = _ensure_orthonormal_triad(ax1_swia, ax2_swia)
    basis_tensor = np.stack([e1, e2, e3], axis=2)  # shape (nt, 3, 3)
    xyz = ts_tensor_xyz(t_arr, basis_tensor)

    # --- Velocity grid (in m/s)
    if vg_1d is None:
        vg_1d = np.linspace(-1000.0, 1000.0, 500) * 1e3
    else:
        vg_1d = np.asarray(vg_1d, dtype=float) * 1e3

    # --- Convert Differential Energy Flux (DEF) → Phase Space Density (PSD)
    psd = vdf.flux_convert(swia_3d, "def2psd")

    # --- Monte Carlo reduction to 1D distribution
    f1D = vdf.reduce(
        psd,
        xyz=xyz,
        dim="1d",
        base="pol",
        vg=vg_1d,
        sc_pot=None,
        n_mc=1000,
    )

    # --- Cleanup SPICE kernels
    sp.kclear()

    return f1D


if __name__ == "__main__":
    # Example usage
    maven.load_maven_spice()
    tint = ["2018-10-18T20:10:00", "2018-10-18T21:10:00"]

    # Load MAVEN SWIA 3D dataset
    swia_3d = maven.get_data(tint, "swia_3d")

    # Define MSO axes
    xmso = np.array((1.0, 0.0, 0.0))
    ymso = np.array((0.0, 1.0, 0.0))

    # Example evaluation time
    t1 = np.datetime64("2018-10-18T20:30:00")

    # Perform reduction
    f1d = reduced_swia_1d(swia_3d, xmso)

    # Mask values below threshold
    f1d.data = np.where(f1d.data > 1e-7, f1d.data, np.nan)

    # Plot result
    ax, cax = plot.plot_spectr(
        None, f1d, cmap="Spectral_r", cscale="log", clim=[1e-7, 1e-4]
    )
    ax.set_ylim(-500.0, 500.0)
    plt.show()

    # Final cleanup
    sp.kclear()

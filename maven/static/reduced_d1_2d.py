import numpy as np
from typing import Union, Iterable
import copy
import py_space_zc
from py_space_zc import normalize, cross, ts_tensor_xyz, ts_vec_xyz
from pyrfu import pyrf

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


def reduced_d1_2d(
    d1: dict,
    time,
    mso_axis1: Union[Iterable[float], np.ndarray],
    mso_axis2: Union[Iterable[float], np.ndarray],
    species: str,
    vg_2d: Union[np.ndarray, Iterable[float], None] = None,
    correct_background = False,
    correct_vsc = False
):
    """
    Reduce a 3D ion velocity distribution (MAVEN STATIC D1) onto a 2D plane
    defined by two MSO-space unit vectors.

    Parameters
    ----------
    d1 : dict-like
        Expected keys:
          - "time"     : array-like time base
          - "H_DEF"    : DEF for H+ (time × energy × angles …)
          - "O_DEF"    : DEF for O+
          - "O2_DEF"   : DEF for O2+
          - "scpot"    : spacecraft potential (V)
          - "sta2mso"  : rotation matrices STA→MSO (time × 3 × 3)
    time : numpy.datetime64 or pandas.Timestamp
        Center time for ±3 s clipping before reduction.
    mso_axis1, mso_axis2 : array-like
        Two non-parallel vectors in MSO. Shapes allowed: (3,), (1,3), or (nt,3).
    species : str
        One of {"H","H+","O","O+","O2","O2+"} (case-insensitive).
    vg_2d : array-like or None
        1D Cartesian velocity grid in km/s. If None, uses [-500, 500] km/s with 200 points.
    correct_background : bool, default False
        If True, apply STATIC D1 background correction.
    correct_vsc : bool, default False
        If True, estimate spacecraft velocity in STATIC frame and apply correction.

    Returns
    -------
    f2D : py_space_zc.vdf-compatible 2D reduced distribution
    """

    # --- prep axes as time series (ts_vec) aligned with D1 time

    if correct_background:
        d1_corr = py_space_zc.maven.static.correct_bkg_d1(d1)
    else:
        d1_corr = copy.deepcopy(d1)

    t_arr = d1_corr["time"]
    ax1_ts = _tile_to_ts_vec(t_arr, np.asarray(mso_axis1), "mso_axis1")
    ax2_ts = _tile_to_ts_vec(t_arr, np.asarray(mso_axis2), "mso_axis2")

    # convert the two MSO axes to STA frame (STATIC native) time-by-time
    # so the reduction basis matches the native VDF frame
    ax1_sta = py_space_zc.maven.static.mso2sta(ax1_ts, d1_corr)  # (nt,3)
    ax2_sta = py_space_zc.maven.static.mso2sta(ax2_ts, d1_corr)  # (nt,3)

    # build right-handed orthonormal triad in STA
    e1, e2, e3 = _ensure_orthonormal_triad(ax1_sta, ax2_sta)

    # pack to (nt,3,3) tensor: columns are basis vectors [e1 e2 e3]
    # shape we want for ts_tensor_xyz is (nt, 3, 3)
    xyz = np.stack([e1, e2, e3], axis=2)  # (nt,3,3)
    xyz = ts_tensor_xyz(t_arr, xyz)

    # velocity grid (m/s)
    if vg_2d is None:
        vg_2d = np.linspace(-500.0, 500.0, 200) * 1e3  # km/s → m/s
    else:
        vg_2d = np.asarray(vg_2d, dtype=float) * 1e3   # km/s → m/s

    # time window ±3 s around center
    tint = pyrf.extend_tint([time, time], [-3, 3])

    # species & DEF→PSD
    s = species.lower()
    if s in {"h", "h+"}:
        vdf_i = py_space_zc.vdf.flux_convert(d1_corr["H_DEF"], "def2psd")
    elif s in {"o", "o+"}:
        vdf_i = py_space_zc.vdf.flux_convert(d1_corr["O_DEF"], "def2psd")
    elif s in {"o2", "o2+"}:
        vdf_i = py_space_zc.vdf.flux_convert(d1_corr["O2_DEF"], "def2psd")
    else:
        raise ValueError(f"Unsupported species: {species}")

    scpot = d1_corr.get("scpot", None)


    # 3. Optional spacecraft velocity correction
    if correct_vsc:
        # Retrieve B-field dataset that includes 'Pmso' (spacecraft position in MSO)
        vsc_mso = py_space_zc.maven.get_vsc_mso(d1_corr["time"])

        # Transform spacecraft velocity from MSO to xyz frame (e1, e2 directions)
        vsc_axis1 = py_space_zc.dot(vsc_mso, e1)
        vsc_axis2 = py_space_zc.dot(vsc_mso, e2)
        vsc_axis1 = np.nanmean(vsc_axis1, axis = 0)
        vsc_axis2 = np.nanmean(vsc_axis2, axis = 0)
    else:
        # Zero spacecraft velocity if no correction
        vsc_axis1 = 0.0
        vsc_axis2 = 0.0


    # reduction on the plane spanned by (e1, e2); basis provided by xyz (STA frame)
    f2D = py_space_zc.vdf.reduce(
        pyrf.time_clip(vdf_i, tint),
        xyz=xyz,
        dim="2d",
        base="cart",
        vg=vg_2d,
        sc_pot=scpot,
        n_mc=1500,  # tune if needed
    )

    f2D = f2D.assign_coords(
        vx=f2D.coords['vx'] + float(vsc_axis1),
        vy=f2D.coords['vy'] + float(vsc_axis2),
    )

    return f2D

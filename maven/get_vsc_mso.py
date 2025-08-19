from typing import Literal
import numpy as np
from pyrfu import pyrf
import py_space_zc.maven as maven
from py_space_zc import resample_time


def get_vsc_mso(
    time: np.ndarray,
    duplicate_policy: Literal["error", "drop", "mean"] = "mean",
    margin_s: float = 5.0,
) -> np.ndarray:
    """
    Estimate MAVEN spacecraft velocity in the MSO (Mars–Solar Orbital) frame by
    finite-differencing the spacecraft position and resampling to the requested timestamps.
    It requires that we can load 1 Hz magnetic field data

    Parameters
    ----------
    time : np.ndarray
        Target time array to which the velocity will be resampled. Must be a 1D array of
        numpy datetime64 (any unit, e.g., datetime64[ns]). Can be of length 1 or more.
    duplicate_policy : {"error", "drop", "mean"}, optional
        How to handle duplicated entries in `time`.
        - "error": raise a ValueError if duplicates are found.
        - "drop":  keep the first occurrence and drop the rest (default).
        - "mean":  collapse duplicates by taking their mean timestamp (rarely useful).
        Note: Duplicated time stamps are a frequent cause of interpolation errors.
    margin_s : float, optional
        Seconds to extend the query interval on both ends when fetching spacecraft position.
        For single-point queries we internally use max(margin_s, 10s) to ensure enough
        ephemeris samples for finite differences. Default is 5 seconds.

    Returns
    -------
    vsc_mso : np.ndarray, shape (len(time), 3)
        Spacecraft velocity vector in MSO coordinates at the requested times.
        Units are km/s (since `Pmso` is in km and time differences are in seconds).
        Values outside the span of the native ephemeris (even with margin) will be NaN.

    Notes
    -----
    - Position source: `maven.get_data(tint, "B")` → `B["Pmso"]` (ts_vec-like), where
      `.data` has shape (N, 3) in km and `.time.data` is numpy datetime64.
    - Velocity estimate: first-order forward difference
          v_i ≈ (r_{i+1} - r_i) / (t_{i+1} - t_i)
      defined at the left timestamps, then linearly resampled to `time`.
    - Guards: sorting, duplicate handling, zero-dt drop, edge extension.
    """

    # --- Validate input time array ----------------------------------------------------
    if time.ndim != 1:
        raise ValueError("`time` must be a 1D numpy array of datetime64.")
    if time.size == 0:
        raise ValueError("`time` must contain at least one timestamp.")

    # Normalize user time array: sort and handle duplicates
    t_user = time.astype("datetime64[ns]")
    order = np.argsort(t_user)
    if not np.all(order == np.arange(t_user.size)):
        t_user = t_user[order]

    unique_mask = np.concatenate(([True], t_user[1:] != t_user[:-1])) if t_user.size > 1 else np.array([True])
    if not np.all(unique_mask):
        if duplicate_policy == "error":
            raise ValueError("Duplicate timestamps found in `time`.")
        elif duplicate_policy == "drop":
            t_user = t_user[unique_mask]
        elif duplicate_policy == "mean":
            t_user = np.unique(t_user)
        else:
            raise ValueError(f"Unknown duplicate_policy: {duplicate_policy}")

    # --- Build an extended interval to fetch position (helps interpolation near edges) --
    # For single-point queries, slightly enlarge the margin to improve robustness.
    is_single_point = (t_user.size == 1)
    local_margin = max(margin_s, 10.0) if is_single_point else margin_s

    # pyrf.extend_tint expects [t0, t1] plus [left_margin, right_margin] in seconds.
    tint = pyrf.extend_tint([t_user[0], t_user[-1]], [-local_margin, local_margin])

    # --- Download magnetic dataset and extract spacecraft position in MSO --------------
    B = maven.get_data(tint, "B")
    Pmso = B["Pmso"]        # ts_vec-like
    pos = Pmso.data         # (N, 3) in km
    tpos = Pmso.time.data   # (N,) numpy datetime64

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise RuntimeError("Unexpected shape for `Pmso.data`; expected (N, 3).")
    if tpos.size < 2:
        # Not enough ephemeris samples to compute finite differences
        raise RuntimeError("Not enough position samples to compute velocity in the requested interval.")

    # --- Compute dt in seconds and finite-difference velocity -------------------------
    tpos_ns = tpos.astype("datetime64[ns]")
    dt = (tpos_ns[1:] - tpos_ns[:-1]) / np.timedelta64(1, "s")  # (N-1,)

    # Guard against non-positive dt
    valid = dt > 0.0
    if not np.any(valid):
        raise RuntimeError("All consecutive position timestamps have non-positive dt; cannot compute velocity.")

    # Slice arrays to keep only valid forward differences
    pos_left = Pmso.data[:-1][valid]
    pos_right = Pmso.data[1:][valid]
    tvel = tpos_ns[:-1][valid]  # left-aligned timestamps for the velocity samples
    dt_valid = dt[valid]

    # Forward-difference velocity in km/s at tvel
    vsc_fd = (pos_right - pos_left) / dt_valid[:, None]  # (M, 3)
    if vsc_fd.size == 0:
        raise RuntimeError("No velocity samples computed (check `dt` and data coverage).")

    # --- Resample velocity to the user-requested times --------------------------------
    # Linear interpolation component-wise; out-of-range times → NaN
    vsc_mso_user = resample_time(tvel, vsc_fd, t_user, "linear")  # (len(t_user), 3)

    # --- Map back to the original order/length (if we sorted or dropped duplicates) ---
    if (not np.all(order == np.arange(time.size))) or (not np.all(unique_mask)):
        v_out = np.full((time.size, 3), np.nan, dtype=float)
        t_sorted = time.astype("datetime64[ns]")[order]
        idx_map = {t: i for i, t in enumerate(t_user)}
        for j, t in enumerate(t_sorted):
            i_user = idx_map.get(t, None)
            if i_user is not None:
                v_out[order[j], :] = vsc_mso_user[i_user, :]
        return v_out

    return vsc_mso_user


if __name__ == "__main__":
    tint = ["2022-10-27T11:41:45", "2022-10-27T11:42:30"]
    B = maven.get_data(tint, "B")
    Pmso = B["Pmso"]
    vsc = get_vsc_mso(Pmso.time.data)
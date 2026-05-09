import numpy as np

from py_space_zc import resample_time, ts_scalar
from py_space_zc.maven import bs_normal


def _as_vec_data(obj, name):
    """Return time and Nx3 data from a ts_vec_xyz-like object."""
    if hasattr(obj, "time") and hasattr(obj, "data"):
        time = np.asarray(obj.time.data)
        data = np.asarray(obj.data, dtype=float)
    else:
        time = None
        data = np.asarray(obj, dtype=float)

    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")

    return time, data


def _get_pmso_from_maven(time):
    """Fetch MAVEN Pmso and resample it to the requested time grid."""
    from pyrfu import pyrf
    import py_space_zc.maven as maven

    t = np.asarray(time).astype("datetime64[ns]")
    if t.ndim != 1 or t.size == 0:
        raise ValueError("B.time must be a non-empty one dimensional datetime64 array.")

    tint = pyrf.extend_tint([t[0], t[-1]], [-5.0, 5.0])
    data = maven.get_data(tint, "B")
    pmso = data["Pmso"]
    return resample_time(pmso.time.data, pmso.data, t, "linear")


def _pmso_to_rm(pmso, r_mars_km):
    """Convert Pmso to Mars radii when values appear to be in km."""
    pmso = np.asarray(pmso, dtype=float)
    finite = np.linalg.norm(pmso[np.all(np.isfinite(pmso), axis=1)], axis=1)
    if finite.size and np.nanmedian(finite) > 100.0:
        return pmso / r_mars_km
    return pmso


def _moving_mean_2d(x, n):
    """Centered running mean for Nx3 arrays, ignoring NaNs."""
    if n is None or n <= 1:
        return x

    n = int(n)
    kernel = np.ones(n, dtype=float) / n
    out = np.full_like(x, np.nan, dtype=float)
    for j in range(x.shape[1]):
        valid = np.isfinite(x[:, j])
        y = np.where(valid, x[:, j], 0.0)
        w = np.convolve(valid.astype(float), kernel, mode="same")
        s = np.convolve(y, kernel, mode="same")
        ok = w > 0.0
        out[ok, j] = s[ok] / w[ok]
    return out


def _line_bs_intersection(pmso_rm, b_xyz):
    """
    Intersect a magnetic field line with the bow shock model used by bs_mpb.

    Returns the nearest shock point and distance along the field line in Rm.
    Invalid or non-connected samples are returned as NaN.
    """
    e = 1.026
    L = 2.081
    x_f = 0.6
    a = e**2 - 1.0
    x_min = -5.0
    x_max = 1.6
    x_tol = 1e-10

    n = pmso_rm.shape[0]
    shock_points = np.full((n, 3), np.nan, dtype=float)
    dist = np.full(n, np.nan, dtype=float)

    b_norm = np.linalg.norm(b_xyz, axis=1)
    valid = (
        np.all(np.isfinite(pmso_rm), axis=1)
        & np.all(np.isfinite(b_xyz), axis=1)
        & (b_norm > 0.0)
    )
    if not np.any(valid):
        return shock_points, dist

    p = pmso_rm[valid]
    u = b_xyz[valid] / b_norm[valid, None]

    qx = p[:, 0] - x_f
    A = a * u[:, 0] ** 2 - u[:, 1] ** 2 - u[:, 2] ** 2
    Bc = 2.0 * a * qx * u[:, 0] - 2.0 * e * L * u[:, 0] - 2.0 * (
        p[:, 1] * u[:, 1] + p[:, 2] * u[:, 2]
    )
    C = a * qx**2 - 2.0 * e * L * qx + L**2 - p[:, 1] ** 2 - p[:, 2] ** 2

    roots = np.full((p.shape[0], 2), np.nan, dtype=float)
    linear = np.isclose(A, 0.0)
    roots[linear, 0] = np.where(np.isclose(Bc[linear], 0.0), np.nan, -C[linear] / Bc[linear])

    quad = ~linear
    disc = Bc[quad] ** 2 - 4.0 * A[quad] * C[quad]
    has_root = disc >= 0.0
    if np.any(has_root):
        idx = np.where(quad)[0][has_root]
        sqrt_disc = np.sqrt(disc[has_root])
        roots[idx, 0] = (-Bc[idx] - sqrt_disc) / (2.0 * A[idx])
        roots[idx, 1] = (-Bc[idx] + sqrt_disc) / (2.0 * A[idx])

    candidate_points = p[:, None, :] + roots[:, :, None] * u[:, None, :]
    candidate_x = candidate_points[:, :, 0]
    in_model = (
        np.isfinite(roots)
        & (candidate_x >= x_min - x_tol)
        & (candidate_x <= x_max + x_tol)
    )
    roots_abs = np.where(in_model, np.abs(roots), np.nan)
    has_candidate = np.any(np.isfinite(roots_abs), axis=1)
    if not np.any(has_candidate):
        return shock_points, dist

    local_choice = np.nanargmin(roots_abs[has_candidate], axis=1)
    local_idx = np.where(has_candidate)[0]
    chosen_s = roots[local_idx, local_choice]
    chosen_points = p[local_idx] + chosen_s[:, None] * u[local_idx]

    global_idx = np.where(valid)[0][local_idx]
    shock_points[global_idx] = chosen_points
    dist[global_idx] = np.abs(chosen_s)

    return shock_points, dist


def _bs_rho(x):
    """Cylindrical bow shock radius used by bs_mpb."""
    e = 1.026
    L = 2.081
    x_f = 0.6
    val = (e**2 - 1.0) * (x - x_f) ** 2 - 2.0 * e * L * (x - x_f) + L**2
    return np.sqrt(np.maximum(val, 0.0))


def _bs_dfdx(x):
    """dF/dx for F = rho_bs(x)^2 - y^2 - z^2."""
    e = 1.026
    L = 2.081
    x_f = 0.6
    return 2.0 * (e**2 - 1.0) * (x - x_f) - 2.0 * e * L


def _find_tangent_points(b_unit, x_min=-5.0, x_max=1.6):
    """Find bow shock points where a line parallel to b_unit is tangent."""
    from scipy.optimize import brentq

    bx = b_unit[0]
    b_perp = np.linalg.norm(b_unit[1:])
    if not np.isfinite(bx) or not np.isfinite(b_perp) or b_perp <= 0.0:
        return np.empty((0, 3), dtype=float)

    e_perp = np.array([0.0, b_unit[1] / b_perp, b_unit[2] / b_perp])
    xs = np.linspace(x_min, x_max, 2001)
    points = []

    for sgn in (-1.0, 1.0):
        g = _bs_dfdx(xs) * bx - 2.0 * sgn * _bs_rho(xs) * b_perp
        good = np.isfinite(g)
        if not np.any(good):
            continue

        for i in np.where(good[:-1] & good[1:])[0]:
            if g[i] == 0.0:
                root = xs[i]
            elif g[i] * g[i + 1] > 0.0:
                continue
            else:
                try:
                    root = brentq(
                        lambda xx: _bs_dfdx(xx) * bx - 2.0 * sgn * _bs_rho(xx) * b_perp,
                        xs[i],
                        xs[i + 1],
                    )
                except ValueError:
                    continue

            rho = _bs_rho(root)
            q = np.array([root, 0.0, 0.0]) + sgn * rho * e_perp
            if not any(np.linalg.norm(q - p) < 1e-6 for p in points):
                points.append(q)

    if not points:
        return np.empty((0, 3), dtype=float)
    return np.vstack(points)


def _foreshock_depth(pmso_rm, b_xyz, connected):
    """
    Signed distance from the spacecraft field line to the closest IMF tangent line.

    Positive values are assigned to field lines that intersect the bow shock,
    negative values to non-connected lines.
    """
    n = pmso_rm.shape[0]
    dif = np.full(n, np.nan, dtype=float)
    b_norm = np.linalg.norm(b_xyz, axis=1)
    valid = (
        np.all(np.isfinite(pmso_rm), axis=1)
        & np.all(np.isfinite(b_xyz), axis=1)
        & (b_norm > 0.0)
    )

    for i in np.where(valid)[0]:
        u = b_xyz[i] / b_norm[i]
        tangent_points = _find_tangent_points(u)
        if tangent_points.size == 0:
            continue
        offsets = np.cross(pmso_rm[i][None, :] - tangent_points, u[None, :])
        depth = np.min(np.linalg.norm(offsets, axis=1))
        dif[i] = depth if connected[i] else -depth

    return dif


def bow_shock_connection(B, Pmso=None, r_mars_km=3393.19, avg_pts=1, return_details=False):
    """
    Compute bow shock theta_bn and DIST for a magnetic field time series.

    Parameters
    ----------
    B : ts_vec_xyz
        Magnetic field in MSO coordinates. The time grid is used for the output.
    Pmso : ts_vec_xyz or ndarray, optional
        MAVEN position in MSO. Units can be km or Rm. If omitted, the function
        fetches MAVEN position from ``maven.get_data`` and resamples it to B.time.
    r_mars_km : float, optional
        Mars radius used to convert km to Rm. Default is 3393.19 km.
    avg_pts : int, optional
        Number of 1 Hz samples in the centered running average applied to B
        before tracing IMF lines. Use 1 for instantaneous B. Values near 11
        reproduce the smoothing described by Meziane et al. for suppressing
        foreshock magnetic fluctuations.
    return_details : bool, optional
        If True, also return the model shock point and normal arrays.

    Returns
    -------
    dict
        ``theta_bn`` and ``DIST`` are ts_scalar objects. The bow shock surface is
        the same conic model used by ``py_space_zc.maven.bs_mpb``. ``theta_bn`` is
        in degrees, with range 0 to 90. ``DIST`` is the distance from the
        spacecraft to the model bow shock along the IMF line, in Rm. ``DIF`` is
        the signed foreshock depth, the distance from the spacecraft field line
        to the nearest IMF tangent line, positive for connected field lines and
        negative otherwise.
    """
    b_time, b_data = _as_vec_data(B, "B")
    if b_time is None:
        raise ValueError("B must be a ts_vec_xyz-like object with a time coordinate.")

    if Pmso is None:
        p_data = _get_pmso_from_maven(b_time)
    else:
        p_time, p_raw = _as_vec_data(Pmso, "Pmso")
        if p_time is not None:
            p_data = resample_time(p_time, p_raw, b_time, "linear")
        else:
            p_data = p_raw

    if p_data.shape[0] != b_data.shape[0]:
        raise ValueError("Pmso and B must have the same number of samples after resampling.")

    b_used = _moving_mean_2d(b_data, avg_pts)
    p_rm = _pmso_to_rm(p_data, r_mars_km)
    shock_points, dist = _line_bs_intersection(p_rm, b_used)
    connected = np.isfinite(dist)
    dif = _foreshock_depth(p_rm, b_used, connected)
    normal = bs_normal(shock_points)
    if normal.ndim == 1:
        normal = normal.reshape(1, 3)

    b_norm = np.linalg.norm(b_used, axis=1)
    n_norm = np.linalg.norm(normal, axis=1)
    dot_bn = np.einsum("ij,ij->i", b_used, normal)
    cos_bn = np.full(b_used.shape[0], np.nan, dtype=float)
    ok = np.isfinite(dot_bn) & (b_norm > 0.0) & (n_norm > 0.0)
    cos_bn[ok] = np.abs(dot_bn[ok]) / (b_norm[ok] * n_norm[ok])
    cos_bn = np.clip(cos_bn, 0.0, 1.0)
    theta_bn = np.degrees(np.arccos(cos_bn))

    out = {
        "theta_bn": ts_scalar(
            np.asarray(b_time),
            theta_bn,
            attrs={
                "name": "theta_bn",
                "units": "deg",
                "description": "Angle between IMF and modeled Mars bow shock normal",
                "avg_pts": int(avg_pts),
            },
        ),
        "DIST": ts_scalar(
            np.asarray(b_time),
            dist,
            attrs={
                "name": "DIST",
                "units": "Rm",
                "description": "Distance from spacecraft to modeled bow shock along IMF line",
                "avg_pts": int(avg_pts),
            },
        ),
        "DIF": ts_scalar(
            np.asarray(b_time),
            dif,
            attrs={
                "name": "DIF",
                "units": "Rm",
                "description": "Signed foreshock depth relative to the nearest IMF tangent line",
                "avg_pts": int(avg_pts),
            },
        ),
    }

    if return_details:
        out["shock_point_mso_rm"] = shock_points
        out["normal_mso"] = normal
        out["B_used_mso"] = b_used

    return out


def bs_theta_bn_dist(B, Pmso=None, r_mars_km=3393.19, avg_pts=1, return_details=False):
    """Alias for :func:`bow_shock_connection`."""
    return bow_shock_connection(B, Pmso, r_mars_km, avg_pts, return_details)

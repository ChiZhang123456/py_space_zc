"""
Adapted from pyrfu's ts_skymap module:
(https://github.com/louis-richard/irfu-python), licensed under the MIT License

Original code licensed under the MIT License.
Modified by Chi Zhang for compatibility with py_space_zc.
"""

import numpy as np
import numba

# %%
@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_pol_1d(
    vdf,          # [n_v, n_phi, n_theta]     : 3D phase space density (skymap) in instrument frame
    v,            # [n_v]                     : velocity bin centers (km/s)
    phi,          # [n_phi]                   : azimuthal angles (radians)
    theta,        # [n_theta] or [n_v, n_theta] : elevation angles (radians), can be 1D or 2D
    d_v,          # [n_v]                     : velocity bin widths
    d_v_m,        # [n_v] or scalar           : offset for lower half-bin width (for Monte Carlo)
    d_phi,        # [n_phi]                   : azimuthal angle bin widths (radians)
    d_theta,      # same shape as theta       : elevation angle bin widths (radians)
    vg_edges,     # [n_vg + 1]                : bin edges of projection grid in velocity space (x-axis)
    d_a_grid,     # [n_vg]                    : azimuthal width per projection bin (for 2D proj.)
    v_lim,        # (2,)                      : limits of projected velocity component (km/s)
    a_lim,        # (2,)                      : angular limits (radians)
    n_mc,         # [n_v, n_phi, n_theta]     : number of Monte Carlo particles per bin
    r_mat         # [3, 3]                    : rotation matrix (instrument frame → projected frame)
):
    """
    Monte-Carlo projection of 3D velocity distribution function (VDF) onto reduced space.

    Parameters
    ----------
    See argument descriptions above.

    Returns
    -------
    f_g : ndarray [n_vg]
        Reduced distribution on the projection grid.
    """
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros(n_vg)  # Output projection result

    for i in numba.prange(n_v):  # Loop over velocity bins
        for j in range(n_ph):    # Loop over azimuth bins
            for k in range(n_th):  # Loop over elevation bins
                n_mc_ijk = n_mc[i, j, k]
                f_ijk = vdf[i, j, k]

                if f_ijk == 0.0:
                    continue

                # ---------------------------------------------------------
                # Select elevation angle and its bin width depending on shape
                # theta: [n_theta] or [n_v, n_theta]
                # d_theta: same shape
                # ---------------------------------------------------------
                if theta.ndim == 1:
                    th = theta[k]
                    dth = d_theta[k]
                else:
                    th = theta[i, k]
                    dth = d_theta[i, k]

                # Compute dτ_ijk: phase space volume element (∝ v^2 * cosθ * Δv * Δϕ * Δθ)
                dtau_ijk = v[i] ** 2 * np.cos(th) * d_v[i] * d_phi[j] * dth
                c_ijk = dtau_ijk / n_mc_ijk  # scaling factor per Monte Carlo particle

                # ---------------------------------------------------------
                # Monte Carlo sampling for this bin
                # ---------------------------------------------------------
                for _ in range(n_mc_ijk):
                    # Random offsets within each bin
                    d_v_mc = -np.random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (np.random.random() - 0.5) * d_phi[j]
                    d_theta_mc = (np.random.random() - 0.5) * dth

                    # Sampled particle parameters
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = th + d_theta_mc

                    # Convert spherical to Cartesian velocities (instrument frame)
                    v_x = v_mc * np.cos(theta_mc) * np.cos(phi_mc)
                    v_y = v_mc * np.cos(theta_mc) * np.sin(phi_mc)
                    v_z = v_mc * np.sin(theta_mc)

                    # Rotate velocity vector into projected frame (v')
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    # Compute projected perpendicular velocity
                    v_perp = np.sqrt(v_y_p ** 2 + v_z_p ** 2)

                    # Compute angle between velocity and x'-axis (in radians)
                    alpha = np.arcsin(v_perp / v_mc)  # Safe arcsin

                    # Apply projection filters (v_perp and alpha)
                    if (
                        v_perp >= v_lim[0] and v_perp < v_lim[1] and
                        alpha >= a_lim[0] and alpha < a_lim[1]
                    ):
                        # Find bin index in projected x-axis grid
                        i_vxg = np.searchsorted(vg_edges[:-2], v_x_p)

                        if i_vxg < n_vg:
                            d_a = d_a_grid[i_vxg]  # angular bin width in projection
                            f_g[i_vxg] += f_ijk * c_ijk / d_a  # contribution
    return f_g


# %%
@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_cart_3d(
    vdf,          # [n_v, n_phi, n_theta]
    v,            # [n_v]
    phi,          # [n_phi]
    theta,        # [n_theta] or [n_v, n_theta]
    d_v,          # [n_v]
    d_v_m,        # [n_v] or scalar
    d_phi,        # [n_phi]
    d_theta,      # same as theta
    vg_edges,     # [n_vg + 1]
    d_a_grid,     # scalar (angular normalization constant)
    v_lim,        # (2,)    -- min/max v_z_p (km/s)
    a_lim,        # (2,)    -- min/max pitch angle (radians)
    n_mc,         # [n_v, n_phi, n_theta]
    r_mat         # [3, 3]
):
    """
    Monte-Carlo projection of 3D VDF into Cartesian grid.

    Parameters
    ----------
    See inline comments for shape and descriptions.

    Returns
    -------
    f_g : ndarray [n_vg, n_vg, n_vg]
        3D velocity distribution interpolated onto Cartesian grid.
    """

    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros((n_vg, n_vg, n_vg))  # Output grid

    for i in numba.prange(n_v):
        for j in range(n_ph):
            for k in range(n_th):
                f_ijk = vdf[i, j, k]
                if f_ijk == 0.0:
                    continue

                n_mc_ijk = n_mc[i, j, k]

                # --- Select theta and d_theta ---
                if theta.ndim == 1:
                    th = theta[k]
                    dth = d_theta[k]
                else:
                    th = theta[i, k]
                    dth = d_theta[i, k]

                # Phase space volume element (∝ v^2 cosθ Δv Δφ Δθ)
                dtau_ijk = v[i] ** 2 * np.cos(th) * d_v[i] * d_phi[j] * dth
                c_ijk = dtau_ijk / n_mc_ijk

                for _ in range(n_mc_ijk):
                    # Monte Carlo perturbation
                    d_v_mc = -np.random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (np.random.random() - 0.5) * d_phi[j]
                    d_theta_mc = (np.random.random() - 0.5) * dth

                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = th + d_theta_mc

                    # Spherical to Cartesian (instrument frame)
                    v_x = v_mc * np.cos(theta_mc) * np.cos(phi_mc)
                    v_y = v_mc * np.cos(theta_mc) * np.sin(phi_mc)
                    v_z = v_mc * np.sin(theta_mc)

                    # Rotate to projection frame
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    # Pitch angle (relative to v_x')
                    alpha = np.arcsin(v_z_p / v_mc)

                    # Selection condition
                    in_vz = v_z_p >= v_lim[0] and v_z_p < v_lim[1]
                    in_alpha = alpha >= a_lim[0] and alpha < a_lim[1]

                    if in_vz and in_alpha:
                        i_vxg = np.searchsorted(vg_edges[:-2], v_x_p)
                        i_vyg = np.searchsorted(vg_edges[:-2], v_y_p)
                        i_vzg = np.searchsorted(vg_edges[:-2], v_z_p)

                        if (i_vxg < n_vg and i_vyg < n_vg and i_vzg < n_vg):
                            f_g[i_vxg, i_vyg, i_vzg] += f_ijk * c_ijk / d_a_grid

    return f_g

# %%
@numba.jit(cache=True, nogil=True, parallel=True, nopython=True)
def _mc_cart_2d(
    vdf,          # [n_v, n_phi, n_theta]
    v,            # [n_v]
    phi,          # [n_phi]
    theta,        # [n_theta] or [n_v, n_theta]
    d_v,          # [n_v]
    d_v_m,        # [n_v] or scalar
    d_phi,        # [n_phi]
    d_theta,      # same shape as theta
    vg_edges,     # [n_vg + 1]
    d_a_grid,     # scalar (normalization factor)
    v_lim,        # (2,) range for v_z' in rotated frame
    a_lim,        # (2,) pitch angle range (radians)
    n_mc,         # [n_v, n_phi, n_theta]
    r_mat         # [3, 3] rotation matrix
):
    """
    Monte Carlo interpolation of 3D VDF onto a 2D Cartesian velocity grid.

    Returns
    -------
    f_g : ndarray [n_vg, n_vg]
        Interpolated distribution in v_x'–v_y' plane.
    """
    n_v, n_ph, n_th = vdf.shape
    n_vg = len(vg_edges) - 1
    f_g = np.zeros((n_vg, n_vg))  # output grid in rotated v_x', v_y'

    for i in numba.prange(n_v):
        for j in range(n_ph):
            for k in range(n_th):
                f_ijk = vdf[i, j, k]
                if f_ijk == 0.0:
                    continue

                n_mc_ijk = n_mc[i, j, k]

                # Select theta/d_theta: support both 1D and 2D
                if theta.ndim == 1:
                    th = theta[k]
                    dth = d_theta[k]
                else:
                    th = theta[i, k]
                    dth = d_theta[i, k]

                # Estimate phase space volume for this bin
                dtau_ijk = v[i]**2 * np.cos(th) * d_v[i] * d_phi[j] * dth
                c_ijk = dtau_ijk / n_mc_ijk

                for _ in range(n_mc_ijk):
                    # Random offset sampling within bin
                    d_v_mc = -np.random.random() * d_v[i] - d_v_m[0]
                    d_phi_mc = (np.random.random() - 0.5) * d_phi[j]
                    d_theta_mc = (np.random.random() - 0.5) * dth

                    # Sampled velocity in spherical coords
                    v_mc = v[i] + d_v_mc
                    phi_mc = phi[j] + d_phi_mc
                    theta_mc = th + d_theta_mc

                    # Convert to Cartesian velocity (instrument frame)
                    v_x = v_mc * np.cos(theta_mc) * np.cos(phi_mc)
                    v_y = v_mc * np.cos(theta_mc) * np.sin(phi_mc)
                    v_z = v_mc * np.sin(theta_mc)

                    # Rotate to projection frame
                    v_x_p = r_mat[0, 0] * v_x + r_mat[1, 0] * v_y + r_mat[2, 0] * v_z
                    v_y_p = r_mat[0, 1] * v_x + r_mat[1, 1] * v_y + r_mat[2, 1] * v_z
                    v_z_p = r_mat[0, 2] * v_x + r_mat[1, 2] * v_y + r_mat[2, 2] * v_z

                    # Compute pitch angle (relative to x' axis)
                    alpha = np.arcsin(v_z_p / v_mc)

                    # Filter: velocity and pitch angle limits
                    in_vz = (v_z_p >= v_lim[0]) and (v_z_p < v_lim[1])
                    in_alpha = (alpha >= a_lim[0]) and (alpha < a_lim[1])

                    if in_vz and in_alpha:
                        # Find bin index
                        i_vxg = np.searchsorted(vg_edges, v_x_p)
                        i_vyg = np.searchsorted(vg_edges, v_y_p)

                        # Bounds check
                        if (i_vxg < n_vg and i_vyg < n_vg):
                            f_g[i_vxg, i_vyg] += f_ijk * c_ijk / d_a_grid

    return f_g
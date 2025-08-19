#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .Monte_Carlo_vdf import _mc_pol_1d as mc_1d
from .Monte_Carlo_vdf import _mc_cart_2d as mc_2d
from .Monte_Carlo_vdf import _mc_cart_3d as mc_3d
import numpy as np

def int_sph_dist(vdf, speed, phi, theta, speed_grid, **kwargs):
    r"""Integrate a spherical distribution function to a line/plane.

    Parameters
    ----------
    vdf : numpy.ndarray
        Phase-space density skymap.
    speed : numpy.ndarray
        Velocity of the instrument bins,
    phi : numpy.ndarray
        Azimuthal angle of the instrument bins.
    theta : numpy.ndarray
        Elevation angle of the instrument bins.
    speed_grid : numpy.ndarray
        Velocity grid for interpolation.
    **kwargs
        Keyw

    Returns
    -------

    """

    # Coordinates system transformation matrix
    xyz = kwargs.get("xyz", np.eye(3))

    # Number of Monte Carlo iterations and how number of MC points is
    # weighted to data.
    n_mc = kwargs.get("n_mc", 10)
    weight = kwargs.get("weight", None)

    # limit on out-of-plane velocity and azimuthal angle
    v_lim = np.array(kwargs.get("v_lim", [-np.inf, np.inf]), dtype=np.float64)
    a_lim = np.array(kwargs.get("a_lim", [-180.0, 180.0]), dtype=np.float64)
    a_lim = np.deg2rad(a_lim)

    # Projection dimension and base
    projection_base = kwargs.get("projection_base", "pol")
    projection_dim = kwargs.get("projection_dim", "1d")

    speed_edges = kwargs.get("speed_edges", None)
    speed_grid_edges = kwargs.get("speed_grid_edges", None)

    # Azimuthal and elevation angles steps. Assumed to be constant
    # if not provided.
    d_phi = np.abs(np.median(np.diff(phi))) * np.ones_like(phi)
    d_phi = kwargs.get("d_phi", d_phi)

    # Check if user provided d_theta
    d_theta_in = kwargs.get("d_theta", None)
    if d_theta_in is not None:
        if np.isscalar(d_theta_in):
            d_theta = d_theta_in * np.ones_like(theta)
        else:
            d_theta = np.array(d_theta_in)
    else:
        # Automatic estimation of d_theta
        if theta.ndim == 1:
            # Case: [n_theta,]
            delta = np.abs(np.median(np.diff(np.sort(theta))))
            d_theta = delta * np.ones_like(theta)

        elif theta.ndim == 2:
            # Case: [n_energy, n_theta]
            d_theta = np.zeros_like(theta)
            for i in range(theta.shape[0]):
                row = theta[i, :]
                delta = np.abs(np.median(np.diff(np.sort(row))))
                d_theta[i, :] = delta


    # azimuthal angle of projection plane
    n_az_g = len(phi)
    d_phi_g = 2 * np.pi / n_az_g
    phi_grid = np.linspace(0, 2 * np.pi - d_phi_g, n_az_g) + d_phi_g / 2
    phi_grid = kwargs.get("phi_grid", phi_grid)

    # Overwrite projection dimension if azimuthal angle of projection
    # plane is not provided. Set the azimuthal angle grid width.
    if phi_grid is not None and projection_dim.lower() in ["2d", "3d"]:
        d_phi_grid = np.median(np.diff(phi_grid))
    else:
        projection_dim = "1d"
        d_phi_grid = 1.0

    # Make sure the transformation matrix is orthonormal.
    x_phat = xyz[:, 0] / np.linalg.norm(xyz[:, 0])  # re-normalize
    y_phat = xyz[:, 1] / np.linalg.norm(xyz[:, 1])  # re-normalize

    z_phat = np.cross(x_phat, y_phat)
    z_phat /= np.linalg.norm(z_phat)
    y_phat = np.cross(z_phat, x_phat)

    r_mat = np.transpose(np.stack([x_phat, y_phat, z_phat]), [1, 0])

    if speed_edges is None:
        d_v = np.hstack([np.diff(speed[:2]), np.diff(speed)])
        d_v_m, d_v_p = [np.diff(speed) / 2.0] * 2
    else:
        d_v_m = speed - speed_edges[:-1]
        d_v_p = speed_edges[1:] - speed
        d_v = d_v_m + d_v_p

    # Speed grid bins edges
    if speed_grid_edges is None:
        speed_grid_edges = np.zeros(len(speed_grid) + 1)
        speed_grid_edges[0] = speed_grid[0] - np.diff(speed_grid[:2]) / 2.0
        speed_grid_edges[1:-1] = speed_grid[:-1] + np.diff(speed_grid) / 2.0
        speed_grid_edges[-1] = speed_grid[-1] + np.diff(speed_grid[-2:]) / 2.0
    else:
        speed_grid = speed_grid_edges[:-1] + np.diff(speed_grid_edges) / 2.0

    if projection_base == "pol":
        d_v_grid = np.diff(speed_grid_edges)
    else:
        mean_diff = np.mean(np.diff(speed_grid))
        msg = "For a cartesian grid, all velocity bins must be equal!!"
        assert (np.diff(speed_grid) / mean_diff - 1 < 1e-2).all(), msg

        d_v_grid = mean_diff

    # Weighting of number of Monte Carlo particles
    n_sum = n_mc * np.sum(vdf != 0)  # total number of Monte Carlo particles
    if weight == "lin":
        n_mc_mat = np.ceil(n_sum / np.sum(vdf) * vdf)
    elif weight == "log":
        n_mc_mat = np.ceil(
            n_sum / np.sum(np.log10(vdf + 1)) * np.log10(vdf + 1),
        )
    else:
        n_mc_mat = np.zeros_like(vdf)
        n_mc_mat[vdf != 0] = n_mc

    n_mc_mat = n_mc_mat.astype(int)

    if projection_base == "cart" and projection_dim == "2d":
        d_a_grid = d_v_grid**2
        f_g = mc_2d(vdf,speed,phi,theta,
                    d_v,d_v_m,d_phi,d_theta,speed_grid_edges,
                    d_a_grid,v_lim,a_lim,n_mc_mat,r_mat,)

    elif projection_base == "cart" and projection_dim == "3d":
        d_a_grid = d_v_grid**3
        f_g = mc_3d(vdf,speed,phi,theta,
                    d_v,d_v_m,d_phi, d_theta,speed_grid_edges,
                    d_a_grid,v_lim,a_lim,n_mc_mat,r_mat,)

    else:
        # Area or line element (primed)
        d_a_grid = speed_grid ** (int(projection_dim[0]) - 1) * d_phi_grid * d_v_grid
        d_a_grid = d_a_grid.astype(np.float64)

        if projection_dim == "1d":
            f_g = mc_1d(vdf,speed,phi,theta,
                d_v,d_v_m,d_phi,d_theta,speed_grid_edges,
                d_a_grid,v_lim,a_lim,n_mc_mat,r_mat,)
        else:
            raise NotImplementedError(
                "2d projection on polar grid is not ready yet!!",
            )

    if projection_dim == "2d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": speed_grid,
            "vy": speed_grid,
            "vx_edges": speed_grid_edges,
            "vy_edges": speed_grid_edges,
        }
    elif projection_dim == "3d" and projection_base == "cart":
        pst = {
            "f": f_g,
            "vx": speed_grid,
            "vy": speed_grid,
            "vz": speed_grid,
            "vx_edges": speed_grid_edges,
            "vy_edges": speed_grid_edges,
            "vz_edges": speed_grid_edges,
        }
    else:
        pst = {"f": f_g, "vx": speed_grid, "vx_edges": speed_grid_edges}

    return pst
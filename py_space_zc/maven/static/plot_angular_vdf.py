import numpy as np
import matplotlib.pyplot as plt
from py_space_zc import plot


def plot_angular_vdf(ax, vdf_data, energyrange=[0, 3000.0], clim=None):
    """
    Plot the angular distribution (phi vs. theta) of a velocity distribution function (VDF)
    integrated over a specified energy range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis object to draw the plot on. If None, a new figure and axis will be created.

    vdf_data : object
        A data object containing:
            - vdf_data.data : ndarray of shape (time, energy, phi, theta)
            - vdf_data.energy.data : energy grid, shape (1, n_energy)
            - vdf_data.phi.data : azimuthal angle grid, shape (1, n_phi)
            - vdf_data.theta.data : polar angle grid, shape (1, n_time, n_theta)

    energyrange : list or tuple of float, optional
        Energy integration range [Emin, Emax] in electron volts (eV). Default is [0, 3000.0].

    clim : list or tuple of float, optional
        Color scale limits [vmin, vmax] for the colormap. If None, autoscaling is used.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plot.

    pcm : matplotlib.collections.QuadMesh
        The pcolormesh object returned by `plot.plot_pcolor`.

    cbar : matplotlib.colorbar.Colorbar
        The colorbar associated with the plot.
    """

    # --- Select energy indices within the specified range ---
    energy = vdf_data.energy.data  # shape: (1, n_energy)
    idx = np.where((energy[0, :] >= energyrange[0]) & (energy[0, :] <= energyrange[1]))[0]

    # --- Adjust phi: shift by 180° + half bin width, wrap to 0–360°, and reorder ---
    phi = vdf_data.phi.data[0, :] - 180.0 - 11.25
    phi[phi <= 0] += 360.0
    phi_new = np.concatenate((phi[8:], phi[0:8]))  # reorder to match instrument orientation

    # --- Adjust theta: center around 0 (i.e., convert from [0, 180] to [-90, 90]) ---
    theta = vdf_data.theta.data[0, 10, :] - 90.0

    # --- Integrate over energy range and average over time ---
    data_all = np.nanmean(vdf_data.data[:, idx, :, :], axis=1)  # mean over energy
    data_all = np.nanmean(data_all, axis=0)  # mean over time
    data_all[data_all <= 0] = np.nan
    data_all = np.concatenate((data_all[8:, :], data_all[0:8, :]), axis=0)

    # --- Create axis if not provided ---
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # --- Plot with custom pcolor function ---
    ax, pcm, cbar = plot.plot_pcolor(ax, phi_new, theta, data_all, cscale='log')

    # --- Add cell borders for visual clarity ---
    pcm.set_edgecolor('k')
    pcm.set_linewidth(0.5)

    # --- Set color scale limits if provided ---
    if clim is not None:
        pcm.set_clim(*clim)
        cbar.update_normal(pcm)

    # --- Axis formatting ---
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.fill_between([0.0, 360.0], -46.0, 46.0, color='black', alpha=0.2)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel(r'$\phi\ (\degree)$', fontsize=12)
    ax.set_ylabel(r'$\theta\ (\degree)$', fontsize=12)

    return ax, pcm, cbar

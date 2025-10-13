import numpy as np
import matplotlib.pyplot as plt
from py_space_zc import maven, plot, norm, ts_scalar
import matplotlib.patches as patches

def show_overview(tint):
    """
    Overview plot of MAVEN (and Tianwen-1) measurements over a given time interval.

    This function generates a publication-quality figure with:
    - Left column: 2 subplots showing MAVEN trajectory in MSO coordinates
        • Top: X–R plane (R = √(Y² + Z²))
        • Bottom: Y–Z plane with Mars as a black circle
    - Right column: 6 vertically stacked data panels:
        1. STATIC C6 H⁺ ion spectrogram
        2. STATIC C6 O⁺ ion spectrogram
        3. STATIC C6 O₂⁺ ion spectrogram
        4. SWIA omnidirectional ion spectra
        5. MAVEN magnetic field Bₓ, Bᵧ, B𝓏 and total field strength |B|
        6. SWEA omnidirectional electron spectra

    Parameters
    ----------
    tint : list of str
        Time interval as [start_time, end_time], in ISO format.
        Example: ["2021-11-19T01:45:00", "2021-11-19T02:20:00"]

    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete figure object, already displayed via `plt.show()`

    Example
    -------
    >>> tint = ["2021-11-19T01:45:00", "2021-11-19T02:20:00"]
    >>> fig = show_overview(tint)
    """

    # -----------------------------------
    # Load MAVEN position and magnetic field data
    # -----------------------------------
    B = maven.get_data(tint, 'B')                       # 1-second cadence B-field
    Pmvn = B["Pmso"].data / 3390.0                      # Normalize to Mars radius
    Rmvn = np.sqrt(Pmvn[:, 1]**2 + Pmvn[:, 2]**2)       # Cylindrical radius
    t_mvn = B["Pmso"].time.data                         # Time array

    # Initialize figure
    fig = plt.figure(figsize=(13, 8.5))

    # ===================================
    # LEFT COLUMN: MAVEN Trajectory
    # ===================================
    _, ax_left = plot.subplot(
        2, 1, fig=fig, hspace=0.5, wspace=0.55,
        bottom=0.1, top=0.93, left=0.07, right=0.3
    )

    # --- Top left: X–R (trajectory & BS/MPB boundaries)
    _, s1, c1 = plot.scatter_time(ax_left[0], Pmvn[:, 0], Rmvn, t_mvn,
                                  cmap='Spectral_r', size=5.0, min_nticks=3)
    maven.bs_mpb(ax_left[0])  # Add bow shock & MPB model
    plot.set_axis(
        ax_left[0], xlim=(-4.0, 2.0), ylim=(0.0, 5.5),
        tick_fontsize=12, label_fontsize=14,
        xlabel=r"$X_{\mathrm{MSO}}\ (R_{\rm M})$",
        ylabel=r"$\sqrt{Y_{\mathrm{MSO}}^2 + Z_{\mathrm{MSO}}^2}$ (R$_\mathrm{M}$)",
        facecolor="white", grid=False
    )
    ax_left[0].set_aspect('equal')
    c1.ax.set_ylabel('')
    plot.adjust_colorbar(ax_left[0], c1, 0.005, 0.8, 0.015)

    # --- Bottom left: Y–Z (with Mars as black disk)
    _, s1, c2 = plot.scatter_time(ax_left[1], Pmvn[:, 1], Pmvn[:, 2], t_mvn,
                                  cmap='Spectral_r', size=5.0, min_nticks=3)
    ax_left[1].add_patch(patches.Circle((0, 0), radius=1, color='black', alpha=0.45, zorder=10))
    plot.set_axis(
        ax_left[1], xlim=(-4.0, 4.0), ylim=(-4.0, 4.0),
        tick_fontsize=12, label_fontsize=14,
        xlabel=r"$Y_{\mathrm{MSO}}\ (R_{\rm M})$", ylabel=r"$Z_{\mathrm{MSO}}\ (R_{\rm M})$",
        facecolor="white", grid=False
    )
    c2.ax.set_ylabel('')
    plot.adjust_colorbar(ax_left[1], c2, 0.005, 0.8, 0.015)

    # ===================================
    # RIGHT COLUMN: 6 stacked panels
    # ===================================
    _, ax_right = plot.subplot(
        6, 1, fig=fig, hspace=0.04, wspace=0.05,
        bottom=0.05, top=0.97, left=0.5, right=0.93, sharex=True
    )

    # --- Panel 1: STATIC C6 H⁺
    ax_h, cax_h = maven.plot_sta_c6(ax_right[0], tint, "H", correct_background=False)
    ax_h.set_xlabel(''); ax_h.set_ylim(5, 30000)
    plot.add_time_title(ax_h, tint, "yyyy/mm/dd HH:MM - HH:MM")
    cax_h.set_ylabel('')

    # --- Panel 2: STATIC C6 O⁺
    ax_o, cax_o = maven.plot_sta_c6(ax_right[1], tint, "O", correct_background=False)
    ax_o.set_xlabel(''); ax_o.set_ylim(5, 30000)

    # --- Panel 3: STATIC C6 O₂⁺
    ax_o2, cax_o2 = maven.plot_sta_c6(ax_right[2], tint, "O2", correct_background=False)
    ax_o2.set_xlabel(''); ax_o2.set_ylim(5, 30000)
    cax_o2.set_ylabel('')

    # --- Panel 4: SWIA omni-directional ion spectra
    ax_swia, cax_swia = maven.plot_swia_omni(ax_right[3], tint)
    ax_swia.set_xlabel(''); ax_swia.set_ylim(20, 30000)
    cax_swia.set_ylabel('')

    # --- Panel 5: SWEA omnidirectional electron spectra
    ax, cax = maven.plot_swea_omni(ax_right[4], tint, clim=[1e5, 5e8])

    # --- Panel 6: MAVEN magnetic field vector & magnitude
    maven.plot_B(ax_right[5], tint)
    B = maven.get_data(tint, 'B')
    Bt = ts_scalar(B["Bmso"].time.data, norm(B["Bmso"].data).reshape(-1))
    plot.plot_line(ax_right[5], Bt, color="grey", linewidth=1.0)
    bmax = np.max(Bt.data)
    ax_right[5].legend(["$B_x$", "$B_y$", "$B_z$", r"$|\mathbf{B}|$"],
                       loc="center right", bbox_to_anchor=(1.123, 0.5),
                       frameon=False, fontsize=12, handlelength=0.5)
    if bmax > 50:
        ax_right[5].set_ylim(-50.0, 50.0)

    # -----------------------------
    # Style all right-column panels
    # -----------------------------
    for ax in ax_right:
        plot.set_axis(ax, fontsize=11, tick_fontsize=11,
                      label_fontsize=12, grid=False)

    return fig

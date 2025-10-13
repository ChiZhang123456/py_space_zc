import numpy as np
import matplotlib.pyplot as plt
from py_space_zc import maven, tianwen_1, plot, norm, ts_scalar

def show_maven_tianwen_data(tint):
    """
    Create a multi-panel figure showing:
      - Left: MAVEN & Tianwen-1 trajectories (X–R and Y–Z), each with a time colorbar.
      - Right: 7 stacked panels: STATIC C6 (H, O, O2), SWIA omni, MAVEN B, Tianwen-1 B,
               and MAVEN+Tianwen-1 clock angle (dot style).

    Parameters
    ----------
    tint : [str, str]
        Time interval, e.g. ["2021-11-19T01:45:00", "2021-11-19T02:20:00"].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure (also displayed via plt.show()).
    """
    # Fetch (ensures data is available; not used directly here but keeps symmetry)
    B = maven.get_data(tint, 'B')
    Btw = tianwen_1.get_data(tint, 'B')

    # Figure
    fig = plt.figure(figsize=(13, 8.5))

    # ---------------- Left column: 2 rows (trajectories) ----------------
    _, ax_left = plot.subplot(
        2, 1, fig=fig, hspace=0.5, wspace=0.55,
        bottom=0.1, top=0.93, left=0.07, right=0.3
    )

    # X–R panel
    _, cbar_xr, _ = tianwen_1.plot_maven_tianwen_xr(ax_left[0], tint)
    plot.set_axis(
        ax_left[0], xlim=(-4.0, 2.0), ylim=(0.0, 5.5),
        tick_fontsize=12, label_fontsize=14,
        xlabel=r"$X_{\mathrm{MSO}}\ (R_{\rm M})$",
        ylabel=r"$\sqrt{Y_{\mathrm{MSO}}^2 + Z_{\mathrm{MSO}}^2}$ (R$_\mathrm{M}$)",
        facecolor="white", grid=False
    )
    ax_left[0].set_aspect('equal')
    cbar_xr.ax.set_ylabel('')
    pos = ax_left[0].get_position()
    cbar_xr.ax.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.015, pos.height - 0.02])

    # Y–Z panel
    _, cbar_yz, _ = tianwen_1.plot_maven_tianwen_yz(ax_left[1], tint)
    plot.set_axis(
        ax_left[1], xlim=(-4.0, 4.0), ylim=(-4.0, 4.0),
        tick_fontsize=12, label_fontsize=14,
        xlabel=r"$Y_{\mathrm{MSO}}\ (R_{\rm M})$", ylabel=r"$Z_{\mathrm{MSO}}\ (R_{\rm M})$",
        facecolor="white", grid=False
    )

    cbar_yz.ax.set_ylabel('')
    pos = ax_left[1].get_position()
    cbar_yz.ax.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.015, pos.height - 0.02])

    # ---------------- Right column: 7 stacked panels ----------------
    _, ax_right = plot.subplot(
        7, 1, fig=fig, hspace=0.04, wspace=0.05,
        bottom=0.05, top=0.97, left=0.5, right=0.93, sharex=True
    )

    # STATIC C6: H
    ax_h, cax_h = maven.plot_sta_c6(ax_right[0], tint, "H", correct_background=False)
    pos = ax_h.get_position()
    cax_h.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.01, pos.height - 0.02])
    ax_h.set_xlabel('')
    if hasattr(cax_h, "set_ylabel"):
        cax_h.set_ylabel('')
    ax_h.set_ylim(5, 30000)
    plot.add_time_title(ax_h, tint, "yyyy/mm/dd HH:MM - HH:MM")
    

    # STATIC C6: O
    ax_o, cax_o = maven.plot_sta_c6(ax_right[1], tint, "O", correct_background=False)
    pos = ax_o.get_position()
    cax_o.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.01, pos.height - 0.02])
    ax_o.set_xlabel('')
    ax_o.set_ylim(5, 30000)

    # STATIC C6: O2 (with background correction)
    ax_o2, cax_o2 = maven.plot_sta_c6(ax_right[2], tint, "O2", correct_background=False)
    pos = ax_o2.get_position()
    cax_o2.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.01, pos.height - 0.02])
    ax_o2.set_xlabel('')
    if hasattr(cax_o2, "set_ylabel"):
        cax_o2.set_ylabel('')
    ax_o2.set_ylim(5, 30000)

    # SWIA omni
    ax_swia, cax_swia = maven.plot_swia_omni(ax_right[3], tint)
    pos = ax_swia.get_position()
    cax_swia.set_position([pos.x1 + 0.005, pos.y0 + 0.01, 0.01, pos.height - 0.02])
    ax_swia.set_xlabel('')
    if hasattr(cax_swia, "set_ylabel"):
        cax_swia.set_ylabel('')
    ax_swia.set_ylim(20, 30000)

    # MAVEN magnetic field
    maven.plot_B(ax_right[4], tint)
    B = maven.get_data(tint,'B')
    Bt = ts_scalar(B["Bmso"].time.data, norm(B["Bmso"].data).reshape(-1))
    plot.plot_line(ax_right[4], Bt, color = "grey", linewidth = 1.0,)
    bmax = np.max(Bt.data)
    ax_right[4].legend(["$B_x$", "$B_y$", "$B_z$", r"$|\mathbf{B}|$"],
               loc="center right", bbox_to_anchor=(1.123, 0.5),
               frameon=False, fontsize=12, handlelength=0.5, ncol=1)
    
    plot.add_text(ax_right[4], 'MAVEN', 0.99, 0.99, color='white', facecolor='grey')
    if bmax>50:
        ax_right[4].set_ylim(-50.0, 50.0)
    
    

    # Tianwen-1 magnetic field
    tianwen_1.plot_B(ax_right[5], tint)
    plot.add_text(ax_right[5], 'Tianwen-1', 0.99, 0.99, color='white', facecolor='grey')
    B = tianwen_1.get_data(tint,'B')
    bmax = np.max(norm(B["Bmso"].data))
    Bt = ts_scalar(B["Bmso"].time.data, norm(B["Bmso"].data).reshape(-1))
    plot.plot_line(ax_right[5], Bt, color = "grey", linewidth = 1.0,)
    if bmax>50:
        ax_right[5].set_ylim(-50.0, 50.0)
        
    # Clock angle (dot style)
    tianwen_1.plot_maven_tianwen_clock_angle(ax_right[6], tint, "dot", markersize=50, linewidth=3)

    for ax in ax_right:
        plot.set_axis(ax, fontsize = 11, tick_fontsize=11, label_fontsize=12)
    return fig

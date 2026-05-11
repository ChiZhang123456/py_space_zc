import numpy as np
import matplotlib.pyplot as plt
from py_space_zc import maven, plot, ts_scalar, norm
import pyrfu.pyrf as pyrf
import matplotlib.patches as patches


def show_maven_data(tint, var):
    """
    Flexible overview plot of selected MAVEN plasma and field data.

    Parameters
    ----------
    tint : list of str
        Time interval as [start_time, end_time] in ISO format.
    var : list of str
        List of variables to plot. Supported values include:
            - "B"            : Magnetic field vector and magnitude
            - "swia_omni"    : SWIA omnidirectional ion spectra
            - "swia_pad"     : SWIA ion pitch angle distribution for all energy range
            - "swea_omni"    : SWEA omnidirectional electron spectra
            - "swea_pad"     : SWEA pitch angle distribution for 100-1000 eV electrons
            - "sta_c6_H"     : STATIC C6 H⁺ ion spectrogram
            - "sta_c6_O"     : STATIC C6 O⁺ ion spectrogram
            - "sta_c6_O2"    : STATIC C6 O₂⁺ ion spectrogram
            - "sta_density"  : STATIC C6 density


    Returns
    -------
    fig : matplotlib.figure.Figure
        The final figure object.
    axs : list of Axes
        List of axes (in order of var).
    """
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



    ncol = 1
    nvar = len(var)
    nrow = nvar
    _, ax_right = plot.subplot(
        nrow, 1, fig=fig, hspace=0.01, wspace=0.05,
        bottom=0.05, top=0.97, left=0.45, right=0.94, sharex=True)

    # Load B data if needed
    if "B" in var:
        B = maven.get_data(tint, "B")
        Bt = pyrf.norm(B["Bmso"])

    for i, v in enumerate(var):
        ax = ax_right[i]

        if v == "B":
            maven.plot_B(ax, tint)
            plot.plot_line(ax, Bt, color="grey", linewidth=1.0)
            ax.legend(["$B_x$", "$B_y$", "$B_z$", r"$|\mathbf{B}|$"],
                      loc="center left", bbox_to_anchor=(1.001, 0.5), fontsize=10, frameon=False)
            bmax = np.nanmax(Bt.data)
            if bmax > 50:
                ax.set_ylim(-50, 50)

        elif v == "swia_omni":
            _, cax = maven.plot_swia_omni(ax, tint)
            cax.set_ylabel('')

        elif v == "swea_omni":
            _, cax = maven.plot_swea_omni(ax, tint, clim = [1e5, 5e8])
            cax.set_ylabel('')

        elif v.startswith("sta_c6_"):
            spc = v.split("_")[-1].upper()  # H, O, O2
            _, cax = maven.plot_sta_c6(ax, tint, spc, correct_background=False)
            cax.set_ylabel('')
            ax.set_ylim(5, 30000)

        elif v == "swia_pad":
            _, cax = maven.plot_swia_pad(ax, tint)
            cax.set_ylabel('')
            ax.set_ylim(0, 180.0)

        elif v == "swea_pad":
            _, cax = maven.plot_swea_pad(ax, tint)
            cax.set_ylabel('')
            ax.set_ylim(0, 180.0)

        elif v == 'sta_density':
            maven.plot_sta_dens(ax, tint, yscale = 'log',)


        else:
            raise ValueError(f"Unknown variable name: {v}")

        # Add time range title to first subplot
        if i == 0:
            plot.add_time_title(ax, tint, "yyyy/mm/dd HH:MM - HH:MM")

        # Set axis style
        plot.set_axis(ax, fontsize=11, tick_fontsize=11,
                      label_fontsize=12, grid=False)

        # Clear xlabel except bottom row
        if i < len(ax_right) - ncol:
            ax.set_xlabel('')

    # Hide unused subplots
    for j in range(nvar, len(ax_right)):
        fig.delaxes(ax_right[j])

    return fig, ax_right[:nvar]


if __name__ == "__main__":
    tint = ["2022-01-24T08:06:30", "2022-01-24T08:09:00"]
    show_maven_data(tint, ['B','swia_omni', 'swea_omni', 'sta_c6_H', 'sta_density'])
    plt.show()
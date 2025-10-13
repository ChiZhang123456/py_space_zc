import numpy as np
import matplotlib.pyplot as plt
from py_space_zc import dot, time_eval, e_vxb, maven, vdf, plot, ts_scalar, ts_vec_xyz
from pyrfu.pyrf import resample, time_clip, extend_tint, normalize
import spiceypy as sp
import py_space_zc.maven.static as static

def vdf_overview(time, species = 'O2', max_vg = 800.0, correct_background = False, correct_vsc = False):
    """
    Generate 6-panel 2D velocity distribution overview at a given time:
    BV-plane, BE-plane, EV-plane, XY, XZ, YZ in MSO coordinates.

    Parameters
    ----------
    time : np.datetime64
        Target time to extract and visualize the distribution.

    Returns
    -------
    None
    """
    sp.kclear()
    tint = extend_tint([time, time], [-9.0, 9.0])
    maven.load_maven_spice()
    B, d1 = maven.load_data(tint, ['B', 'static_d1'])

    # Canonical MSO unit vectors
    N = len(d1['time'])
    xmso = np.zeros((N, 3), dtype=np.float32)
    ymso = np.zeros((N, 3), dtype=np.float32)
    zmso = np.zeros((N, 3), dtype=np.float32)
    xmso[:, 0] = 1.0
    ymso[:, 1] = 1.0
    zmso[:, 2] = 1.0

    # Normalized unit vectors in MSO
    bmso = normalize(resample(B['Bmso'], d1['H_DEF']))
    moment = static.moments_d1(d1, 'H', correct_background=correct_background, correct_vsc = correct_vsc)
    vmso = moment['V']
    emso = normalize(e_vxb(vmso, bmso, flag='vxb'))
    vmso = normalize(e_vxb(emso, bmso, flag='exb'))

    # Reduced 2D VDFs
    vg = np.linspace(-max_vg, max_vg, 200)
    f2d_xy = static.reduced_d1_2d(d1, xmso, ymso, species, vg_2d = vg,
                                  correct_background=correct_background, correct_vsc = correct_vsc)
    f2d_xz = static.reduced_d1_2d(d1, xmso, zmso, species, vg_2d = vg,
                                  correct_background=correct_background, correct_vsc = correct_vsc)
    f2d_yz = static.reduced_d1_2d(d1, ymso, zmso, species, vg_2d = vg,
                                  correct_background=correct_background, correct_vsc = correct_vsc)
    f2d_bv = static.reduced_d1_2d(d1, bmso.data, vmso.data, species, vg_2d = vg,
                                  correct_background=correct_background, correct_vsc = correct_vsc)
    f2d_be = static.reduced_d1_2d(d1, bmso.data, emso.data, species, vg_2d = vg,
                                  correct_background=correct_background, correct_vsc = correct_vsc)
    f2d_ev = static.reduced_d1_2d(d1, emso.data, vmso.data, species, vg_2d = vg,
                                  correct_background=correct_background, correct_vsc = correct_vsc)

    # Grid
    vx = f2d_bv.vx.data
    vy = f2d_bv.vy.data

    # Dot product projections
    vb = dot(moment['V'].data, bmso.data)
    ve = dot(moment['V'].data, emso.data)
    vv = dot(moment['V'].data, vmso.data)

    # MSO components
    vx_mso = moment['V'].data[:, 0]
    vy_mso = moment['V'].data[:, 1]
    vz_mso = moment['V'].data[:, 2]

    # Interpolate to target time
    vb_val = time_eval(ts_scalar(d1['time'], vb), time).data
    ve_val = time_eval(ts_scalar(d1['time'], ve), time).data
    vv_val = time_eval(ts_scalar(d1['time'], vv), time).data
    vx_val = time_eval(ts_scalar(d1['time'], vx_mso), time).data
    vy_val = time_eval(ts_scalar(d1['time'], vy_mso), time).data
    vz_val = time_eval(ts_scalar(d1['time'], vz_mso), time).data


    label_map = {
        "h": r"$\mathrm{H}^+$", "h+": r"$\mathrm{H}^+$", "p": r"$\mathrm{H}^+$",
        "he": r"$\mathrm{He}^+$", "he+": r"$\mathrm{He}^+$", "he++": r"$\mathrm{He}^+$",
        "o": r"$\mathrm{O}^+$", "o+": r"$\mathrm{O}^+$",
        "o2": r"$\mathrm{O}_2^+$", "o2+": r"$\mathrm{O}_2^+$",
        "co2": r"$\mathrm{CO}_2^+$", "co2+": r"$\mathrm{CO}_2^+$",
    }

    def nice_label(species: str) -> str:
        k = species.lower()
        return label_map[k]
    label = nice_label(species)


    # ========== Plotting ==========
    fig, axs = plot.subplot(2, 3, figsize=(13, 8),
                            hspace=0.35, wspace=0.35, bottom=0.1)

    def plot_plane(ax, data, labelx, labely, add_colorbar=False):
        data = time_eval(data, time).data.T
        data = np.where(data < 1e-12, np.nan, data)
        _, pcm, cax = plot.plot_pcolor(ax, vx, vy, data, cscale='log', colorbar = add_colorbar)
        if add_colorbar:
            plot.adjust_colorbar(ax, cax, pad=0.01, height_ratio=0.6, width=0.015)
            cax.ax.set_ylabel(r'Phase Space Density [s$^3$/m$^6$]')
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)
        ax.plot([-1000, 1000], [0, 0], 'gray', linestyle='--')
        ax.plot([0, 0], [-1000, 1000], 'gray', linestyle='--')
        plot.set_axis(ax, xlim=(-max_vg, max_vg), ylim=(-max_vg, max_vg),
                      fontsize=11, tick_fontsize=12, label_fontsize=12, facecolor = 'black')

    plot_plane(axs[0], f2d_bv, r'$V_B$ (km/s)', r'$V_{E \times B}$ (km/s)')
    plot_plane(axs[1], f2d_be, r'$V_B$ (km/s)', r'$V_E$ (km/s)')
    plot_plane(axs[2], f2d_ev, r'$V_E$ (km/s)', r'$V_{E \times B}$ (km/s)', add_colorbar=True)
    plot_plane(axs[3], f2d_xy, r'$V_X$ (km/s)', r'$V_Y$ (km/s)')
    plot_plane(axs[4], f2d_xz, r'$V_X$ (km/s)', r'$V_Z$ (km/s)')
    plot_plane(axs[5], f2d_yz, r'$V_Y$ (km/s)', r'$V_Z$ (km/s)', add_colorbar=True)

    # Add title
    plot.add_time_title(axs[1], np.datetime64(time), "yyyy-mm-dd HH:MM:SS", fontsize=15, fontweight='bold')
    plot.add_text(axs[0], 'STATIC VDF Overview of ' + label, 0.1, 1.15,
                  va = 'bottom', ha = 'left',fontsize=15, fontweight='bold')
    # Add stars
    axs[0].plot(vb_val, vv_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[1].plot(vb_val, ve_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[2].plot(ve_val, vv_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[3].plot(vx_val, vy_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[4].plot(vx_val, vz_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[5].plot(vy_val, vz_val, color='k', marker='*', linestyle='None', markersize=12)


    for ax in axs:
        plot.add_text(ax, label, 0.98, 0.98, fontsize=12, color = 'white')

    plt.show()

    return fig, axs


if __name__ == '__main__':
    time = ["2022-01-24T08:06:30"]
    vdf_overview(time[0], max_vg = 100.0)
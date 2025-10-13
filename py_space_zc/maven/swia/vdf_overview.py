import numpy as np
import matplotlib.pyplot as plt
from py_space_zc import dot, time_eval, e_vxb, maven, vdf, plot, ts_scalar, ts_vec_xyz
from pyrfu.pyrf import resample, time_clip, extend_tint, normalize
import spiceypy as sp


def vdf_overview(time):
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
    tint = extend_tint([time, time], [-10.0, 10.0])
    maven.load_maven_spice()
    B, swia_3d = maven.load_data(tint, ['B', 'swia_3d'])

    # Normalized unit vectors in MSO
    bmso = normalize(resample(B['Bmso'], swia_3d))
    moment = maven.swia.moment_swia_3d(swia_3d)
    emso = normalize(e_vxb(moment['Vmso'], B["Bmso"], flag='vxb'))
    vmso = normalize(e_vxb(emso, B["Bmso"], flag='exb'))

    # Canonical MSO unit vectors
    N = len(swia_3d.time.data)
    xmso = np.zeros((N, 3), dtype=np.float32)
    ymso = np.zeros((N, 3), dtype=np.float32)
    zmso = np.zeros((N, 3), dtype=np.float32)
    xmso[:, 0] = 1.0
    ymso[:, 1] = 1.0
    zmso[:, 2] = 1.0

    # Reduced 2D VDFs
    f2d_xy = maven.swia.reduced_swia_2d(swia_3d, xmso, ymso)
    f2d_xz = maven.swia.reduced_swia_2d(swia_3d, xmso, zmso)
    f2d_yz = maven.swia.reduced_swia_2d(swia_3d, ymso, zmso)
    f2d_bv = maven.swia.reduced_swia_2d(swia_3d, bmso.data, vmso.data)
    f2d_be = maven.swia.reduced_swia_2d(swia_3d, bmso.data, emso.data)
    f2d_ev = maven.swia.reduced_swia_2d(swia_3d, emso.data, vmso.data)

    # Grid
    vx = f2d_bv.vx.data
    vy = f2d_bv.vy.data

    # Dot product projections
    vb = dot(moment['Vmso'], bmso.data)
    ve = dot(moment['Vmso'], emso.data)
    vv = dot(moment['Vmso'], vmso.data)

    # MSO components
    vx_mso = moment['Vmso'].data[:, 0]
    vy_mso = moment['Vmso'].data[:, 1]
    vz_mso = moment['Vmso'].data[:, 2]

    # Interpolate to target time
    vb_val = time_eval(ts_scalar(swia_3d.time.data, vb), time).data
    ve_val = time_eval(ts_scalar(swia_3d.time.data, ve), time).data
    vv_val = time_eval(ts_scalar(swia_3d.time.data, vv), time).data
    vx_val = time_eval(ts_scalar(swia_3d.time.data, vx_mso), time).data
    vy_val = time_eval(ts_scalar(swia_3d.time.data, vy_mso), time).data
    vz_val = time_eval(ts_scalar(swia_3d.time.data, vz_mso), time).data

    # ========== Plotting ==========
    fig, axs = plot.subplot(2, 3, figsize=(13, 8), hspace=0.35, wspace=0.35, bottom=0.1)

    def plot_plane(ax, data, labelx, labely, add_colorbar=False):
        data = time_eval(data, time).data.T
        data = np.where(data < 1e-8, np.nan, data)
        _, pcm, cax = plot.plot_pcolor(ax, vx, vy, data, cscale='log')
        if add_colorbar:
            plot.adjust_colorbar(ax, cax, pad=0.01, height_ratio=0.6, width=0.015)
            cax.ax.set_ylabel(r'Phase Space Density [s$^3$/m$^6$]')
        else:
            cax.remove()
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)
        ax.plot([-1000, 1000], [0, 0], 'gray', linestyle='--')
        ax.plot([0, 0], [-1000, 1000], 'gray', linestyle='--')
        plot.set_axis(ax, xlim=(-600, 600), ylim=(-600, 600),
                      fontsize=11, tick_fontsize=12, label_fontsize=12, facecolor = 'black')

    plot_plane(axs[0], f2d_bv, r'$V_B$ (km/s)', r'$V_{E \times B}$ (km/s)')
    plot_plane(axs[1], f2d_be, r'$V_B$ (km/s)', r'$V_E$ (km/s)')
    plot_plane(axs[2], f2d_ev, r'$V_E$ (km/s)', r'$V_{E \times B}$ (km/s)', add_colorbar=True)
    plot_plane(axs[3], f2d_xy, r'$V_X$ (km/s)', r'$V_Y$ (km/s)')
    plot_plane(axs[4], f2d_xz, r'$V_X$ (km/s)', r'$V_Z$ (km/s)')
    plot_plane(axs[5], f2d_yz, r'$V_Y$ (km/s)', r'$V_Z$ (km/s)', add_colorbar=True)

    # Add title
    plot.add_time_title(axs[1], np.datetime64(time), "yyyy-mm-dd HH:MM:SS", fontsize=15, fontweight='bold')
    plot.add_text(axs[0], 'SWIA VDF Overview', 0.1, 1.15,
                  va = 'bottom', ha = 'left',fontsize=15, fontweight='bold')
    # Add stars
    axs[0].plot(vb_val, vv_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[1].plot(vb_val, ve_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[2].plot(ve_val, vv_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[3].plot(vx_val, vy_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[4].plot(vx_val, vz_val, color='k', marker='*', linestyle='None', markersize=12)
    axs[5].plot(vy_val, vz_val, color='k', marker='*', linestyle='None', markersize=12)

    plt.show()

    return fig, axs


if __name__ == '__main__':
    time = ["2018-10-18T20:25:00"]
    vdf_overview(time[0])
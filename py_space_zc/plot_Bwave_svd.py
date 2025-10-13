from py_space_zc import method, plot, ts_scalar
import matplotlib.pyplot as plt
from py_space_zc import background_B
from .gyro_information import gyro_information
import pyrfu.pyrf as pyrf
import numpy as np

def plot_Bwave_svd(Bwave, window_length = 20.0, overlap = 10.0,freq_range = [0.05, 16.0]):
    Bbgd = background_B(Bwave, window_length=window_length, overlap=overlap)
    Bt = pyrf.norm(Bbgd)
    v, f, r = gyro_information(Bt, 100.0, 'H+')
    fc = ts_scalar(Bt.time.data, f)
    flh = 42.9 * fc

    wave_res = method.SVD_B(Bwave,
                            window_length=window_length,
                            overlap=overlap,
                            freq_range=freq_range)
    B_all = wave_res['Bperp'] + wave_res['Bpara']
    ratio = wave_res['Bpara'] / wave_res['Bperp']
    fig, axs = plot.subplot(6, 1, figsize=(10.0, 9.5), hspace=0.02, top=0.95, right=0.85, bottom=0.07, sharex=True)
    plot.plot_line(axs[0], Bwave, linewidth = 1.0,)
    axs[0].set_ylabel('B(nT)')
    axs[0].legend(["$B_x$", "$B_y$", "$B_z$"],
                    loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, fontsize=12, handlelength=0.5, ncol=1)

    _, c1 = plot.plot_spectr(axs[1], B_all, yscale='log', cscale='log', cmap='Spectral_r')
    _, c3 = plot.plot_spectr(axs[2], ratio, yscale='log', cscale='log', cmap='coolwarm', clim=[0.1, 10.0])
    _, c4 = plot.plot_spectr(axs[3], wave_res['theta'], yscale='log', cscale='lin', cmap='coolwarm', clim=[0.0, 90.0])
    _, c5 = plot.plot_spectr(axs[4], wave_res['ellipticity'], yscale='log', cscale='lin', cmap='coolwarm',
                             clim=[-1.0, 1.0])
    _, c6 = plot.plot_spectr(axs[5], wave_res['planarity'], yscale='log', cscale='lin', cmap='Spectral_r',
                             clim=[0.0, 1.0])
    plot.add_text(axs[1], 'B', 0.99, 0.98, color='white', facecolor='gray', fontsize=13)
    plot.add_text(axs[2], 'Bpara/Bperp', 0.99, 0.98, color='white', facecolor='gray', fontsize=13)
    plot.add_text(axs[3], '$\\theta$', 0.99, 0.98, color='white', facecolor='gray', fontsize=13)
    plot.add_text(axs[4], 'Ellipticity', 0.99, 0.98, color='white', facecolor='gray', fontsize=13)
    plot.add_text(axs[5], 'Planarity', 0.99, 0.98, color='white', facecolor='gray', fontsize=13)

    plot.adjust_colorbar(axs[1], c1, pad=0.01, height_ratio=0.6, width=0.015)
    plot.adjust_colorbar(axs[2], c3, pad=0.01, height_ratio=0.6, width=0.015)
    plot.adjust_colorbar(axs[3], c4, pad=0.01, height_ratio=0.6, width=0.015)
    plot.adjust_colorbar(axs[4], c5, pad=0.01, height_ratio=0.6, width=0.015)
    plot.adjust_colorbar(axs[5], c6, pad=0.01, height_ratio=0.6, width=0.015)
    c1.set_ylabel("$B^2$" + "\n" + "[nT$^2$ Hz$^{-1}$]")
    c3.set_ylabel("Ratio")
    c4.set_ylabel("$\\theta_k$" + "\n" + "[$^\\circ$]")

    tint = [Bwave.time.data[0], Bwave.time.data[-1]]
    plot.add_time_title(axs[0], tint, fontsize=14)

    plot.set_axis(axs[0], fontsize=12, tick_fontsize=12, label_fontsize=13, )
    for ax in axs[1:]:
        plot.plot_line(ax, fc, color = 'black', linewidth = 1.5)
        plot.plot_line(ax, flh, color='black', linewidth=1.5)
        ax.set_ylabel('Freq(Hz)')
        ax.set_yticks(np.array((0.001, 0.01, 0.1, 1, 10, 100)))
        plot.set_axis(ax, fontsize=12, tick_fontsize=12, label_fontsize=13, ylim = (freq_range[0], freq_range[1]))

    return fig, axs


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, LogNorm
from py_space_zc.plot import plot_line, plot_spectr, set_axis, add_colorbar
from py_space_zc import ts_spectr, norm, ts_vec_xyz, ts_scalar, plot, vdf, time_eval, vex
import matplotlib.cm as cm
from typing import Union, Iterable, Tuple, Optional
from pyrfu import pyrf


#%% Show the Bvso data (0.25Hz)
def plot_B(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    tint_new = pyrf.extend_tint(tint, [-5.0, 5.0])
    B = vex.get_data(tint_new, "B")
    plot_line(ax, B)
    ax.set_ylabel(r"$B_{\mathrm{VSO}}\ (\mathrm{nT})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax
#%% Show the Bvso data (1Hz)
def plot_B_1s(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    tint_new = pyrf.extend_tint(tint, [-5.0, 5.0])
    B = vex.get_data(tint_new, "B_1s")
    plot_line(ax, B)
    ax.set_ylabel(r"$B_{\mathrm{VSO}}\ (\mathrm{nT})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax
#%% Show the ELS omni countdata
def plot_els_omni(ax, tint, cmap = 'Spectral_r', clim=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    tint_new = pyrf.extend_tint(tint, [-8.0, 8.0])
    eomni = vex.get_data(tint_new, "els_omni")

    if clim is None:
        ax, cax = plot_spectr(ax, eomni, yscale="log", cscale="log",
                              cmap=cmap,)
    else:
        ax, cax = plot_spectr(ax, eomni, yscale="log", cscale="log",
                              cmap=cmap, clim=clim)

    ax.set_ylabel("$ELS$\n$E_e$ [eV]")
    cax.set_ylabel("Count")
    ax.set_ylim([1, 30000])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.7, width=0.010)
    return ax, cax

#%% Show the ELS epad data
def plot_els_pad(ax, tint, energyrange = [100.0, 1000.0],cmap = 'Spectral_r', option="norm"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
   
    epad = vex.get_data(tint, 'els_pad')
    def linspace_datetime64(t_start, t_end, num):
        t0 = t_start.astype('datetime64[ns]').astype('float64')
        t1 = t_end.astype('datetime64[ns]').astype('float64')
        return np.linspace(t0, t1, num).astype('datetime64[ns]')
    time = epad.time.data
    tnew = linspace_datetime64(time[0], time[-1], int(len(time)/3))
    data_new = ts_scalar(tnew, np.zeros(len(tnew)))
    epad = pyrf.resample(epad, data_new)
    
    # Merge over energy range to get PAD(time, pitchangle)
    pad_range = vdf.pitchangle_merge_energy(
        epad,
        energyrange=energyrange,
        option = option
    )

    # Plot PAD
    if option == "norm":
        ax, cax = plot_spectr(ax, pad_range, cmap=cmap, clim=[0, 2])
        cax.set_ylabel("Norm(PSD)")
    else:
        ax, cax = plot_spectr(ax, pad_range, cscale="log", cmap=cmap)
        cax.set_ylabel("PSD\n[$s^3 / m^6$]")

    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.7, width=0.010)
    ax.set_ylabel("ELS PAD [$^\\circ$]")
    ax.set_ylim([0, 180])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))

    e1, e2 = energyrange
    if e2 < 1000:
        label_str = f"{e1:.0f}–{e2:.0f} eV"
    else:
        label_str = f"{e1/1000:.1f}–{e2/1000:.1f} keV"
    plot.add_text(ax, label_str, 0.98, 0.98, color = 'white', facecolor = 'gray')
    return ax



if __name__ == "__main__":
    tint = ["2006-05-15T01:00:00", "2006-05-15T01:05:00"]
    plot_els_omni(None, tint, )
    plt.show()
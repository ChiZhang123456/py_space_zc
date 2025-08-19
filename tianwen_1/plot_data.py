# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:47:05 2024

@author: Win
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, LogNorm
from pyrfu import pyrf
from py_space_zc.plot import plot_line
from py_space_zc.plot import plot_spectr
import py_space_zc.tianwen_1 as tw
from py_space_zc import pad_split_energy  # Ensure this is implemented correctly
import matplotlib.cm as cm

def plot_B(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    B = tw.get_data(tint, "B")
    plot_line(ax, B["Bmso"])
    ax.set_ylabel(r"$B_{\mathrm{MSO}}\ (\mathrm{nT})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax


def plot_minpa_mod1_omni(ax, tint, species, clim=None):
    """
    Plot Tianwen-1 Minpa mode 1 omnidirectional differential energy flux (DEF)
    for a selected ion species over a given time interval.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis will be created.
    tint : tuple of str or np.datetime64
        Time interval (start, end) for data extraction.
    species : str
        Ion species, e.g., 'H+', 'He+', 'O+', 'O2+', 'CO2+'.
    clim : list or tuple of float, optional
        Color scale limits for the plot. Default is [1e5, 1e9].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The axis with the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Get data
    data = tw.minpa_omni(tint, species)

    # Use default clim if not provided
    if clim is None:
        clim = [1e5, 1e9]

    ax, cax = plot_spectr(ax, data, yscale="log", cscale="log",
                          clim=clim, cmap='Spectral_r')

    sp = species.lower()
    # Define mass ranges and display label
    if sp in ["h", "h+", "p"]:
        label = "$\\mathrm{H}^+$"
    elif sp in ["he", "he+", "he++"]:
        label = "$\\mathrm{He}^+$"
    elif sp in ["o", "o+"]:
        label = "$\\mathrm{O}^+$"
    elif sp in ["o2", "o2+"]:
        label = "$\\mathrm{O}_2^+$"
    elif sp in ["co2", "co2+"]:
        label = "$\\mathrm{CO}_2^+$"
    else:
        raise ValueError(f"Unsupported species: {species}")

    ax.set_ylabel("$\\mathrm{MINPA}$\n$E_i$ [eV]")
    ax.set_xlabel("Time")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    cax.set_ylabel("DEF\n[keV/(cm$^2$ s sr keV)]")

    # Add label
    bbox_props = dict(boxstyle='round,pad=0.3',
                      facecolor='black', edgecolor='none',
                      alpha=0.6)
    ax.text(0.97, 0.97, label, transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)

    return ax, cax



if __name__ == "__main__":
    tint = ["2022-10-18T13:08", "2022-10-18T13:20"]
    plot_minpa_mod1_omni(None, tint,'H')
    plt.show()

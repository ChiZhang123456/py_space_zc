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
from pyrfu.plot import plot_line
from py_space_zc import plot_spectr
import py_space_zc.maven as maven
from py_space_zc import pad_split_energy  # Ensure this is implemented correctly
import matplotlib.cm as cm

def plot_B(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    B = maven.get_data(tint, "B")
    plot_line(ax, B["Bmso"])
    ax.set_ylabel(r"$B_{\mathrm{MSO}}\ (\mathrm{nT})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax


def plot_B_high(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    B = maven.get_data(tint, "B_high")
    plot_line(ax, B["Bmso"])
    ax.set_ylabel(r"$B_{\mathrm{MSO}}\ (\mathrm{nT})$")
    ax.legend(["$B_x$", "$B_y$", "$B_z$", r"$|\mathbf{B}|$"],
              loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1,
              handlelength=2, frameon=False)
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    ax.text(0.02, 0.95, 'MAVEN', transform=ax.transAxes, fontsize=12,
            ha='left', va='top', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='gray', edgecolor='none'))
    return ax


def plot_B_mse(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    B = maven.get_data(tint, "Bmse")
    plot_line(ax, B["Bmse"])
    ax.set_ylabel(r"$B_{\mathrm{MSE}}\ (\mathrm{nT})$")
    ax.legend(["$B_x$", "$B_y$", "$B_z$", r"$|\mathbf{B}|$"],
              loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1,
              handlelength=2, frameon=False)
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    ax.text(0.02, 0.95, 'MAVEN', transform=ax.transAxes, fontsize=12,
            ha='left', va='top', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='gray', edgecolor='none'))
    return ax


def plot_swia_omni(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    swia = maven.get_data(tint, "swia_omni")
    ax, cax = plot_spectr(ax, swia["omni"], yscale="log", cscale="log", cmap="Spectral_r")
    ax.set_ylabel("$SWIA\quad E_i$ [eV]")
    cax.set_ylabel("DEF\n[kev/(cm$^2$ s sr keV)]")
    ax.set_ylim([25, 20000])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax

def plot_sta_c6(ax, tint, species):
    """
    Plot MAVEN STATIC C6 omnidirectional differential energy flux (DEF)
    for a selected ion species over a given time interval.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis will be created.
    tint : tuple of str or np.datetime64
        Time interval (start, end) for data extraction.
    species : str
        Ion species, e.g., 'H+', 'He+', 'O+', 'O2+', 'CO2+'.

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
    c6 = maven.static.extract_data_c6(tint, species)
    ax, cax = plot_spectr(ax, c6, yscale="log", cscale="log",
                            clim=[1e4, 5*1e6], cmap='Spectral_r')

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

    ax.set_ylabel("$\\mathrm{STATIC}\\quad E_i$ [eV]")
    ax.set_xlabel("Time")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    cax.set_ylabel("DEF\n[kev/(cm$^2$ s sr keV)]")
    
    # Add label
    bbox_props = dict(boxstyle='round,pad=0.3',
                      facecolor='black', edgecolor='none',
                      alpha=0.6)
    ax.text(0.97, 0.97, label, transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)
    return ax, cax

def plot_swea_omni(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    swea = maven.get_data(tint, "swea_omni")
    ax, cax = plot_spectr(ax, swea, yscale="log", cscale="log", cmap="Spectral_r")
    ax.set_ylabel("$SWEA\quad E_e$ [eV]")
    cax.set_ylabel("DEF\n[kev/(cm$^2$ s sr keV)]")
    ax.set_ylim([3, 3000])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax


def plot_swea_epad(ax, tint, option="norm"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    swea_pad = maven.get_data(tint, "swea_pad")
    pad_range = pad_split_energy(swea_pad["pad"], [100.0, 1000.0])
    mean_pad = np.nanmean(pad_range.values, axis=1)
    norm_pad_data = pad_range.values / mean_pad[:, None]
    norm_pad = pad_range.copy()
    norm_pad.data = norm_pad_data

    if option == "norm":
        ax, cax = plot_spectr(ax, norm_pad, cmap="Spectral_r", clim=[0, 2])
        cax.set_ylabel("Norm(DEF)")
    else:
        ax, cax = plot_spectr(ax, pad_range, cscale="log", cmap="Spectral_r")
        cax.set_ylabel("DEF\n[kev/(cm$^2$ s sr keV)]")

    ax.set_ylabel("$\\theta$ [$^\\circ$]")
    ax.set_ylim([0, 180])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    ax.text(0.05, 0.1, "0.1-1 keV", transform=ax.transAxes)
    return ax


if __name__ == "__main__":
    tint = ["2015-09-21T13:00", "2015-09-21T16:00"]
    plot_sta_c6(None, tint,'H')
    plt.show()

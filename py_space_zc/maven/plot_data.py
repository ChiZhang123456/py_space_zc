"""
@author: Chi Zhang
This script defines plotting and data processing utilities for analyzing
MAVEN spacecraft observations in the Martian plasma environment.
Functions include:


- `plot_B`, `plot_B_high`, `plot_B_mse`: Plot magnetic field vectors in MSO or MSE coordinates.
- `plot_swia_omni`: Display SWIA omnidirectional ion energy spectra.
- `plot_swia_pad`: Show SWIA pitch angle distribution (PAD).
- `plot_swia_reduced_2d`: Reduce 3D SWIA ion distribution onto a 2D MSO-plane.
- `plot_swia_vpar_perp`: Visualize SWIA ion VDF in Vpara-Vperp coordinates.
- `plot_sta_c6`: Plot STATIC C6 omnidirectional differential energy flux for selected species.
- `plot_sta_dens`: Display STATIC density time series for H⁺, O⁺, O₂⁺.
- `plot_d1_reduced_1d`: 1D velocity distribution reduction for STATIC D1 data.
- `plot_d1_reduced_2d`: 2D velocity distribution reduction for STATIC D1 data.
- `plot_d1_flux`: Compute and show STATIC ion flux time series.
- `plot_swea_omni`: SWEA omnidirectional electron spectra.
- `plot_swea_pad`: SWEA pitch angle distribution over a defined energy range.
- 'plot_swea_resample_pad': SWEA resampled pitch angle distribution over a defined energy range.
- `plot_crustal_field_map`: Draw Mars crustal magnetic field contour map from Langlais model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, LogNorm
from py_space_zc.plot import plot_line, plot_spectr, set_axis, add_colorbar
import py_space_zc.maven as maven
from py_space_zc import ts_spectr, norm, ts_vec_xyz, ts_scalar, plot, vdf, time_eval
import matplotlib.cm as cm
from typing import Union, Iterable, Tuple, Optional
from pyrfu import pyrf

# --- Global Figure Configuration ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['mathtext.fontset'] = 'stix'
# Fix the "missing minus sign" hyphen issue in Times New Roman
plt.rcParams['axes.unicode_minus'] = False
# Optional: Set global font size if needed
# plt.rcParams['font.size'] = 12

#%% Show the Bmso data (1Hz)
def plot_B(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    tint_new = pyrf.extend_tint(tint, [-5.0, 5.0])
    B = maven.get_data(tint_new, "B")
    plot_line(ax, B["Bmso"])
    ax.set_ylabel(r"$B_{\mathrm{MSO}}\ (\mathrm{nT})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax

#%% Show the Bmso data (32Hz)
def plot_B_high(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    tint_new = pyrf.extend_tint(tint, [-5.0, 5.0])
    B = maven.get_data(tint_new, "B_high")
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

#%% Show the Bmse data (1Hz)
def plot_B_mse(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    tint_new = pyrf.extend_tint(tint, [-5.0, 5.0])
    B = maven.get_data(tint_new, "Bmse")
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

#%% Show the SWIA omni data
def plot_swia_omni(ax, tint, cmap = 'Spectral_r', clim=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    tint_new = pyrf.extend_tint(tint, [-8.0, 8.0])
    swia = maven.get_data(tint_new, "swia_omni")

    if clim is None:
        ax, cax = plot_spectr(ax, swia["omni_flux"], yscale="log", cscale="log",
                              cmap=cmap,)
    else:
        ax, cax = plot_spectr(ax, swia["omni_flux"], yscale="log", cscale="log",
                              cmap=cmap, clim=clim)
    ax.set_ylabel("SWIA" + "\n" + "E [eV]", fontname='Times New Roman', fontsize=12)
    cax.set_ylabel("DEF\n[keV/(cm$^2$ s sr keV)]")
    ax.set_ylim([25, 20000])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.6, width=0.010)
    return ax, cax

#%% Show the SWIA epad data
def plot_swia_pad(ax, tint, energyrange = [0.0, 10000.0],cmap='Spectral_r', option=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    tint_new = pyrf.extend_tint(tint, [-16.0, 16.0])
    swia_pad = maven.swia.get_pad(tint_new, delta_angles=22.5)
    # Merge over energy range to get PAD(time, pitchangle)
    pad_range = vdf.pitchangle_merge_energy(
        swia_pad,
        energyrange=energyrange,
        option = option
    )

    # Plot PAD
    if option == "norm":
        ax, cax = plot_spectr(ax, pad_range, cmap=cmap, clim=[0, 2])
        cax.set_ylabel("Norm(DEF)")
    else:
        ax, cax = plot_spectr(ax, pad_range, cscale="log", cmap=cmap)
        cax.set_ylabel("DEF\n[keV/(cm$^2$·s·sr·keV)]")

    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.6, width=0.015)
    ax.set_ylabel("SWIA PAD [$^\\circ$]")
    ax.set_ylim([0, 180])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    e1, e2 = energyrange
    if e2 < 1000:
        label_str = f"{e1:.0f}–{e2:.0f} eV"
    else:
        label_str = f"{e1/1000:.1f}–{e2/1000:.1f} keV"
    plot.add_text(ax, label_str, 0.98, 0.98, color = 'white', facecolor = 'gray')
    return ax

#%% Show the SWIA reduced 1d data
def plot_swia_reduced_1d(
        ax: Optional[plt.Axes],
        swia_3d,
        mso_axis: Union[np.ndarray, Iterable[float]],
        vg: Union[np.ndarray, Iterable[float], None] = None,
        *,
        cmap: str = "Spectral_r",
        ylabel: str = "V (km/s)",
        clim: Optional[Tuple[float, float]] = None,
):
    """
    Plot a 1D reduction of MAVEN SWIA ion VDF onto the direction of mso_axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If None, a new figure/axes will be created.
    swia_3d :
    mso_axis : array-like
        vectors in MSO coordinates. Shapes can be (3,), (1,3),
        or (nt,3) aligned with d1['time'].
    species : str
        One of {"H","H+","O","O+","O2","O2+"} (case-insensitive).
    vg : array-like or None
        1D Cartesian velocity grid for reduction. Provided in km/s.
    correct_background : bool, default False
        If True, apply STATIC D1 background correction.
    correct_vsc : bool, default False
        If True, estimate spacecraft velocity in STATIC frame and apply correction.
    cmap : str, default "Spectral_r"
        Colormap name.
    ylabel : str
        Axis labels (displayed in km/s).
    clim : (vmin, vmax) or None
        Color limits passed to the plotting function.

    Returns
    -------
    ax, cax : (matplotlib.axes.Axes, matplotlib.axes.Axes)
        Axes for the image and its colorbar.
    """

    # 1) Ensure we have an Axes to draw on
    # 1) Ensure we have an Axes to draw on
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # 2) Normalize vg_2d units: reducer expects m/s; display uses km/s
    vg_arr = None
    xylim_kms = (-600.0, 600.0)  # default display limits in km/s
    if vg is not None:
        vg_arr = np.asarray(vg, dtype=float)
        xylim_kms = (float(np.nanmin(vg_arr)), float(np.nanmax(vg_arr)))

    # 3) Perform reduction
    f1d = maven.swia.reduced_swia_1d(
        swia_3d, mso_axis, vg_1d=vg_arr,)

    ax, pcm, cax = plot.plot_pcolor(ax, f1d.time.data, f1d.vx.data, f1d.data, 
                                    cscale='log',
                                    clim=clim, cmap = cmap)
    cax.remove()
    cax = plot.add_colorbar(ax, pcm, size_ratio = 0.8, 
                            thickness_ratio = 0.015, pad = 0.005 )
    xmin, xmax = xylim_kms
    ax.plot([f1d.time.data[0], f1d.time.data[-1]], [0, 0], 
            color="grey", linestyle="--", linewidth=1)

    # 7) Axes limits and labels
    ax.set_ylim(*xylim_kms)
    ax.set_xlim(f1d.time.data[0], f1d.time.data[-1])
    ax.set_ylabel(ylabel)
    ax.grid(True, color='grey', linewidth=0.2)
    cax.set_label(r"$\mathrm{s\,m^{-4}}$")
    # Add label
    bbox_props = dict(boxstyle='round,pad=0.3',
                      facecolor='black', edgecolor='none',
                      alpha=0.6)
    ax.text(0.97, 0.97, "SWIA", transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)
    return ax, cax


# %% Show the SWIA 2d-reduced vdf data

def plot_swia_reduced_2d(
        ax: Optional[plt.Axes],
        swia_3d,
        t_2d,  # center time (numpy.datetime64 / pandas.Timestamp)
        mso_axis1: Union[np.ndarray, Iterable[float]],
        mso_axis2: Union[np.ndarray, Iterable[float]],
        vg_2d: Union[np.ndarray, Iterable[float], None] = None,
        *,
        cmap: str = "Spectral_r",
        xlabel: str = r"$v_1$ (km/s)",
        ylabel: str = r"$v_2$ (km/s)",
        clim: Optional[Tuple[float, float]] = None,
):
    """
    Plot a 2D reduction of MAVEN SWIA ion VDF onto the plane spanned by
    two MSO-space unit vectors (mso_axis1, mso_axis2).

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If None, a new figure/axes will be created.
    swia_3d:
    t_2d : datetime-like
        Center time. The reduction routine will internally clip/average around
        this time (typically ±3 s).
    mso_axis1, mso_axis2 : array-like
        Two non-parallel vectors in MSO coordinates. Shapes can be (3,), (1,3),
        or (nt,3) aligned with swia_3d.time.data.
    vg_2d : array-like or None
        1D Cartesian velocity grid for reduction. Provided in km/s (converted to m/s).
    cmap : str, default "Spectral_r"
        Colormap name.
    xlabel, ylabel : str
        Axis labels (displayed in km/s).
    clim : (vmin, vmax) or None
        Color limits passed to the plotting function.

    Returns
    -------
    ax, cax : (matplotlib.axes.Axes, matplotlib.axes.Axes)
        Axes for the image and its colorbar.
    """

    # 1) Ensure we have an Axes to draw on
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # 2) Normalize vg_2d units: reducer expects m/s; display uses km/s
    if vg_2d is not None:
        vg_arr = np.asarray(vg_2d, dtype=float)
        xylim_kms = (float(np.nanmin(vg_arr)), float(np.nanmax(vg_arr)))
    else:
        vg_arr = np.linspace(-600.0,600.0,600)
        xylim_kms = (-600.0, 600.0)  # default display limits in km/s

    # 3) Perform reduction
    f2d = maven.swia.reduced_swia_2d(
        swia_3d, mso_axis1, mso_axis2, vg_2d=vg_arr,)

    # 4) Average over time dimension if present
    f2d_for_plot = time_eval(f2d, t_2d)
    t_vals = np.array(f2d_for_plot.time.data)
    
    
    try:
        if hasattr(f2d, "dims") and ("time" in getattr(f2d, "dims", [])):
            f2d_for_plot = f2d.mean(dim="time")
        elif hasattr(f2d, "ndim") and f2d.ndim >= 3:
            f2d_for_plot = f2d.mean(axis=0)
    except Exception:
        pass

    # 5) Draw spectrogram
    ax, cax = plot_spectr(ax, f2d_for_plot, cscale = 'log', 
                          cmap = cmap, clim = clim)

    # 6) Crosshair at v1=0, v2=0 in km/s
    xmin, xmax = xylim_kms
    ax.plot([xmin, xmax], [0, 0], color="grey", linestyle="--", linewidth=1)
    ax.plot([0, 0], [xmin, xmax], color="grey", linestyle="--", linewidth=1)

    # 7) Axes limits and labels
    ax.set_xlim(*xylim_kms)
    ax.set_ylim(*xylim_kms)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color='grey', linewidth=0.2)

    # 8) Colorbar label
    if hasattr(cax, "set_ylabel"):
        cax.set_ylabel(r"$\mathrm{s^3\,m^{-6}}$")

    # 9) Add title

    if t_vals.size == 1:
        t_str = np.datetime_as_string(t_vals, unit='s')[11:]
    else:
        t_min_str = np.datetime_as_string(t_vals.min(), unit='s')[11:]
        t_max_str = np.datetime_as_string(t_vals.max(), unit='s')[11:]
        t_str = t_min_str

    label = 'SWIA'
    # Add label
    bbox_props = dict(boxstyle='round,pad=0.3',
                      facecolor='black', edgecolor='none',
                      alpha=0.6)
    ax.text(0.97, 0.97, label, transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)
    ax.text(0.97, 0.1, t_str, transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)
    return ax, cax

# %% Show the SSWIA vpar-vperp plane
def plot_swia_vpar_perp(ax, time,
                        cmap: str = "Spectral_r",
                        xlabel: str = r"$v_1$ (km/s)",
                        ylabel: str = r"$v_2$ (km/s)",
                        clim: Optional[Tuple[float, float]] = None,):
    # 1) Ensure we have an Axes to draw on
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    tint = pyrf.extend_tint([time, time], [-16.0, 16.0])
    B, swia_3d = maven.load_data(tint, ['B', 'swia_3d'])
    swia_par_perp = maven.swia.vpar_perp_plane(swia_3d, B['Bmso'])
    vpar = swia_par_perp.vpar.data
    vperp = swia_par_perp.vperp.data
    par_perp = time_eval(swia_par_perp, time)
    par_perp = np.where(par_perp < 1e-8, np.nan, par_perp)
    _, pcm, cax = plot.plot_pcolor(ax, vpar, vperp, par_perp,
                                   cscale='log', clim=clim)
    plot.adjust_colorbar(ax, cax, pad=0.01, height_ratio=0.8, width=0.01)
    plot.set_axis(ax, xlim=(-800.0, 800.0), ylim=(0.0, 800.0),
                  fontsize=11, tick_fontsize=12, label_fontsize=12)
    cax.ax.set_ylabel(r'Phase Space Density [s$^3$/m$^6$]')
    ax.plot([0.0, 0.0], [-1000.0, 1000.0], color='gray', linestyle='--')
    ax.set_xlabel(r'$v_\parallel$ (km/s)')
    ax.set_ylabel(r'$|v_\perp|$ (km/s)')
    plot.add_text(ax, 'SWIA',0.99, 0.98, color = 'white', facecolor = 'gray', fontsize = 13)
    return ax, cax


# %% Show the STATIC C6 omni data
def plot_sta_c6(ax, tint, species, clim=None, cmap = 'Spectral_r',correct_background=False):
    """
    Plot MAVEN STATIC C6 omnidirectional differential energy flux (DEF)
    for one or multiple ion species over a given time interval.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis will be created.
    tint : tuple of (str | np.datetime64)
        Time interval (start, end) for data extraction.
    species : str or Sequence[str]
        Ion species, e.g., 'H+', 'He+', 'O+', 'O2+', 'CO2+'.
        You may also pass multiple species as a list/tuple, e.g. ['O+','O2+'],
        or as a comma-separated string, e.g. 'O+,O2+'.
        When multiple species are provided, their DEF are summed elementwise.
    clim : (float, float), optional
        Color scale limits for the plot. Default is [1e4, 5e7].
    correct_background : bool, optional
        Whether to apply STATIC background correction in extraction.

    Returns
    -------
    ax, cax
        Matplotlib axes for the spectrogram and colorbar axis.
    """

    # --- Parse species argument into a clean list of strings ---
    if isinstance(species, str):
        # allow comma-separated string like "O+,O2+"
        sp_list = [s.strip() for s in species.split(",") if s.strip()]
    else:
        sp_list = list(species)

    if len(sp_list) == 0:
        raise ValueError("`species` must contain at least one ion species.")

    # --- Create axis if needed ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # --- Extract first species to set the reference grid ---
    tint_new = pyrf.extend_tint(tint, [-16.0, 16.0])
    c6_ref = maven.static.extract_data_c6(tint_new, sp_list[0], correct_background)
    time_ref = np.asarray(c6_ref.time.data)
    energy_ref = np.asarray(c6_ref.energy.data)
    DEF_sum = np.array(c6_ref.data, copy=True)  # avoid modifying the original

    # --- For any additional species, extract and sum on the same grid ---
    for sp in sp_list[1:]:
        c6_i = maven.static.extract_data_c6(tint_new, sp, correct_background)

        # Basic consistency checks (shapes and grids)
        time_i = np.asarray(c6_i.time.data)
        energy_i = np.asarray(c6_i.energy.data)
        DEF_i = np.asarray(c6_i.data)

        # Sum DEF elementwise
        DEF_sum += DEF_i

    # --- Build a combined ts_spectr for plotting ---
    c6_combined = ts_spectr(
        time=time_ref,
        ener=energy_ref,
        data=DEF_sum,
        comp_name="energy",
        attrs={"UNITS": "keV/(cm^2 s sr keV)"})

    # --- Plot spectrogram ---
    if clim is None:
        clim = [1e4, 1e7]

    ax, cax = plot_spectr(ax, c6_combined, yscale="log", cscale="log",
                          clim = clim, cmap=cmap)

    # --- Build a nice label (TeX) ---
    label_map = {
        "h": r"$\mathrm{H}^+$", "h+": r"$\mathrm{H}^+$", "p": r"$\mathrm{H}^+$",
        "he": r"$\mathrm{He}^+$", "he+": r"$\mathrm{He}^+$", "he++": r"$\mathrm{He}^+$",
        "o": r"$\mathrm{O}^+$", "o+": r"$\mathrm{O}^+$",
        "o2": r"$\mathrm{O}_2^+$", "o2+": r"$\mathrm{O}_2^+$",
        "co2": r"$\mathrm{CO}_2^+$", "co2+": r"$\mathrm{CO}_2^+$",
    }

    def nice_label(sp_raw: str) -> str:
        k = sp_raw.lower()
        if k not in label_map:
            raise ValueError(f"Unsupported species: {sp_raw}")
        return label_map[k]

    if len(sp_list) == 1:
        label = nice_label(sp_list[0])
    else:
        label = " + ".join(nice_label(s) for s in sp_list)

    # --- Axes cosmetics ---
    ax.set_ylabel(r"$\mathrm{STATIC}$" "\n" r"$E_i$ [eV]")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    ax.set_yticks([1e1, 1e2, 1e3, 1e4])
    cax.set_ylabel(r"DEF" "\n" r"[keV/(cm$^2$ s sr keV)]")

    # --- Corner label box ---
    bbox_props = dict(facecolor='white',
                      edgecolor='none', alpha=1.0,)
    ax.text(0.97, 0.95, label, transform=ax.transAxes, color='black',
            ha='right', va='top', family='serif',fontsize=14, bbox=bbox_props)
    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.6, width=0.010)
    return ax, cax


#%% Show the STATIC c8 deflection map
def plot_sta_c8(ax, tint, energyrange = [0.0, 10000.0],cmap='Spectral_r'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    tint_new = pyrf.extend_tint(tint, [-16.0, 16.0])
    sta_c8 = maven.get_data(tint_new, 'static_c8')
    # Merge over energy range to get PAD(time, pitchangle)
    energymat = np.tile(sta_c8['energy'][:, :, None], (1, 1, 16))
    mask = (energymat >= energyrange[0]) & (energymat <= energyrange[1])
    data = np.nansum(np.where(mask, sta_c8['DEF'], np.nan), axis=1)
    dangle = np.linspace(-42.0, 42.0, num=16)
    data = ts_spectr(sta_c8['time'], dangle, data, comp_name = "theta" )
    ax, cax = plot_spectr(ax, data, cscale="log", cmap=cmap, edgecolor="k", linewidth=0.5)
    cax.set_ylabel("DEF\n[keV/(cm$^2$·s·sr·keV)]")
    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.6, width=0.015)
    ax.set_ylim(-90.0, 90.0)
    ax.set_yticks([-90.0, -45.0, 0, 45.0, 90.0])
    ax.set_ylabel(r'Deflection Angles ($^\circ$)')
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    plot.set_axis(ax, fontsize=10, tick_fontsize=12, label_fontsize=13)
    ax.axhline(y=45, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=-45, color='black', linestyle='--', linewidth=1)
    # ax.fill_between([np.datetime64(tint[0]), np.datetime64(tint[1])],
    #                 -45.0, 45.0, color='grey', alpha=0.2)
    return ax


#%% Show the STATIC density data
def plot_sta_dens(ax,
                  tint,
                  yscale: str = "linear",
                  markersize: float = 7.0,
                  linewidth: float = 1.0,):
    """
    Plot MAVEN STATIC ion densities (H+, O+, O2+) over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis to plot on. If None, a new figure and axis are created.
    tint : list of str
        Time interval in ISO format, e.g., ["2023-09-17T05:27:00", "2023-09-17T05:35:00"].
    yscale : str, optional
        Y-axis scale, either "linear" or "log". Default is "linear".
    markersize : float, optional
        Size of markers for data points. Default is 7.0.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the density plot.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    tint_new = pyrf.extend_tint(tint, [-16.0, 16.0])
    dens = maven.get_data(tint_new, "static_density")
    plot_line(ax, dens["nH"],  label="H",  color="black", linestyle="-", linewidth = linewidth,
                   marker=".", markersize=markersize)
    plot_line(ax, dens["nO"],  label="O",  color="blue",  linestyle="-", linewidth = linewidth,
                   marker=".", markersize=markersize)
    plot_line(ax, dens["nO2"], label="O2", color="red",   linestyle="-", linewidth = linewidth,
                   marker=".", markersize=markersize)
    ax.set_ylabel(r"$N\;(\mathrm{cm}^{-3})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))

    if yscale == "log":
        ax.set_yscale("log")
    elif yscale != "linear":
        raise ValueError(f"Unsupported yscale: {yscale}. Use 'linear' or 'log'.")

    ax.legend(
        [r"$\mathrm{H}^+$", r"$\mathrm{O}^+$", r"$\mathrm{O}_2^+$"],
        loc="center left",              # Align legend box to center right
        bbox_to_anchor=(1.01, 0.5),       # Place anchor point just outside the right axis
        frameon=False,
        fontsize=12,
        handlelength=0.5,
        ncol=1)
    return ax

#%% Show the STATIC d1 1d-reduced vdf data
def plot_d1_reduced_1d(
    ax: Optional[plt.Axes],
    d1: dict,
    mso_axis: Union[np.ndarray, Iterable[float]],
    species: str,
    vg: Union[np.ndarray, Iterable[float], None] = None,
    correct_background: bool = False,
    correct_vsc: bool = False,
    *,
    cmap: str = "Spectral_r",
    ylabel: str = r"$v_2$ (km/s)",
    clim: Optional[Tuple[float, float]] = None,
):
    """
    Plot a 1D reduction of MAVEN STATIC D1 ion VDF onto the direction of mso_axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If None, a new figure/axes will be created.
    d1 : dict
        STATIC D1 dataset with keys like 'time', 'H_DEF'/'O_DEF'/'O2_DEF',
        'scpot', and 'sta2mso' (used internally by reduced_d1_2d).
    mso_axis : array-like
        vectors in MSO coordinates. Shapes can be (3,), (1,3),
        or (nt,3) aligned with d1['time'].
    species : str
        One of {"H","H+","O","O+","O2","O2+"} (case-insensitive).
    vg : array-like or None
        1D Cartesian velocity grid for reduction. Provided in km/s (converted to m/s).
    correct_background : bool, default False
        If True, apply STATIC D1 background correction.
    correct_vsc : bool, default False
        If True, estimate spacecraft velocity in STATIC frame and apply correction.
    cmap : str, default "Spectral_r"
        Colormap name.
    ylabel : str
        Axis labels (displayed in km/s).
    clim : (vmin, vmax) or None
        Color limits passed to the plotting function.

    Returns
    -------
    ax, cax : (matplotlib.axes.Axes, matplotlib.axes.Axes)
        Axes for the image and its colorbar.
    """

    # 1) Ensure we have an Axes to draw on
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # 2) Normalize vg_2d units: reducer expects m/s; display uses km/s
    vg_ms_for_reduce = None
    xylim_kms = (-500.0, 500.0)  # default display limits in km/s
    if vg is not None:
        vg_arr = np.asarray(vg, dtype=float)
        vg_ms_for_reduce = vg_arr * 1e3
        xylim_kms = (float(np.nanmin(vg_arr)), float(np.nanmax(vg_arr)))

    # 3) Perform reduction
    f1d = maven.static.reduced_d1_1d(
        d1, mso_axis, species, vg = vg_arr,
        correct_background = correct_background,
        correct_vsc = correct_vsc
    )

    # 5) Draw spectrogram
    ax, pcm, cax = plot.plot_pcolor(ax, f1d.time.data, vg, f1d.data, cscale='log',
                                     clim=clim, cmap = cmap)
    cax.remove()
    cax = plot.add_colorbar(ax, pcm, size_ratio = 0.8, thickness_ratio = 0.015, pad = 0.005 )
    # 6) Crosshair at v1=0, v2=0 in km/s
    xmin, xmax = xylim_kms
    ax.plot([f1d.time.data[0], f1d.time.data[-1]], [0, 0], color="grey", linestyle="--", linewidth=1)

    # 7) Axes limits and labels
    ax.set_ylim(*xylim_kms)
    ax.set_xlim(f1d.time.data[0], f1d.time.data[-1])
    ax.set_ylabel(ylabel)
    ax.grid(True, color='grey', linewidth=0.2)
    cax.set_label(r"$\mathrm{s\,m^{-4}}$")

    sp = species.lower()
    label_map = {
        "h": "$\\mathrm{H}^+$", "h+": "$\\mathrm{H}^+$", "p": "$\\mathrm{H}^+$",
        "he": "$\\mathrm{He}^+$", "he+": "$\\mathrm{He}^+$", "he++": "$\\mathrm{He}^+$",
        "o": "$\\mathrm{O}^+$", "o+": "$\\mathrm{O}^+$",
        "o2": "$\\mathrm{O}_2^+$", "o2+": "$\\mathrm{O}_2^+$",
        "co2": "$\\mathrm{CO}_2^+$", "co2+": "$\\mathrm{CO}_2^+$"
    }
    if sp not in label_map:
        raise ValueError(f"Unsupported species: {species}")
    label = label_map[sp]
    # Add label
    bbox_props = dict(boxstyle='round,pad=0.3',
                      facecolor='black', edgecolor='none',
                      alpha=0.6)
    ax.text(0.97, 0.97, label, transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)
    return ax, cax

#%% Show the STATIC d1 2d-reduced vdf data

def plot_d1_reduced_2d(
    ax: Optional[plt.Axes],
    d1: dict,
    t_2d,  # center time (numpy.datetime64 / pandas.Timestamp)
    mso_axis1: Union[np.ndarray, Iterable[float]],
    mso_axis2: Union[np.ndarray, Iterable[float]],
    species: str,
    vg_2d: Union[np.ndarray, Iterable[float], None] = None,
    correct_background: bool = False,
    correct_vsc: bool = False,
    *,
    cmap: str = "Spectral_r",
    xlabel: str = r"$v_1$ (km/s)",
    ylabel: str = r"$v_2$ (km/s)",
    clim: Optional[Tuple[float, float]] = None,
):
    """
    Plot a 2D reduction of MAVEN STATIC D1 ion VDF onto the plane spanned by
    two MSO-space unit vectors (mso_axis1, mso_axis2) and show the ±3 s time range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If None, a new figure/axes will be created.
    d1 : dict
        STATIC D1 dataset with keys like 'time', 'H_DEF'/'O_DEF'/'O2_DEF',
        'scpot', and 'sta2mso' (used internally by reduced_d1_2d).
    t_2d : datetime-like
        Center time. The reduction routine will internally clip/average around
        this time (typically ±3 s).
    mso_axis1, mso_axis2 : array-like
        Two non-parallel vectors in MSO coordinates. Shapes can be (3,), (1,3),
        or (nt,3) aligned with d1['time'].
    species : str
        One of {"H","H+","O","O+","O2","O2+"} (case-insensitive).
    vg_2d : array-like or None
        1D Cartesian velocity grid for reduction. Provided in km/s (converted to m/s).
    correct_background : bool, default False
        If True, apply STATIC D1 background correction.
    correct_vsc : bool, default False
        If True, estimate spacecraft velocity in STATIC frame and apply correction.
    cmap : str, default "Spectral_r"
        Colormap name.
    xlabel, ylabel : str
        Axis labels (displayed in km/s).
    clim : (vmin, vmax) or None
        Color limits passed to the plotting function.

    Returns
    -------
    ax, cax : (matplotlib.axes.Axes, matplotlib.axes.Axes)
        Axes for the image and its colorbar.
    """

    # 1) Ensure we have an Axes to draw on
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # 2) Normalize vg_2d units: reducer expects m/s; display uses km/s
    if vg_2d is not None:
        vg_arr = np.asarray(vg_2d, dtype=float)
        vg_ms_for_reduce = vg_arr
        xylim_kms = (float(np.nanmin(vg_arr)), float(np.nanmax(vg_arr)))
    else:
        vg_arr = np.linspace(-500.0, 500.0, 500)
        xylim_kms = (-500.0, 500.0)  # default display limits in km/s
        
    # 3) Perform reduction
    f2d = maven.static.reduced_d1_2d(
        d1, mso_axis1, mso_axis2, species, vg_2d=vg_arr,
        correct_background = correct_background,
        correct_vsc = correct_vsc
    )

    # 4) Average over time dimension if present
    f2d_for_plot = time_eval(f2d, t_2d)

    # 5) Draw spectrogram
    ax, cax = plot_spectr(ax, f2d_for_plot, cscale='log', cmap=cmap, clim=clim)

    # 6) Crosshair at v1=0, v2=0 in km/s
    xmin, xmax = xylim_kms
    ax.plot([xmin, xmax], [0, 0], color="grey", linestyle="--", linewidth=1)
    ax.plot([0, 0], [xmin, xmax], color="grey", linestyle="--", linewidth=1)

    # 7) Axes limits and labels
    ax.set_xlim(*xylim_kms)
    ax.set_ylim(*xylim_kms)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color='grey', linewidth=0.2)
    cax.set_ylabel(r"$\mathrm{s^2\,m^{-5}}$")

    # 9) Add title
    t_vals = np.array(f2d_for_plot.time)
    t_str = np.datetime_as_string(t_vals, unit='s')[11:]

    sp = species.lower()
    label_map = {
        "h": "$\\mathrm{H}^+$", "h+": "$\\mathrm{H}^+$", "p": "$\\mathrm{H}^+$",
        "he": "$\\mathrm{He}^+$", "he+": "$\\mathrm{He}^+$", "he++": "$\\mathrm{He}^+$",
        "o": "$\\mathrm{O}^+$", "o+": "$\\mathrm{O}^+$",
        "o2": "$\\mathrm{O}_2^+$", "o2+": "$\\mathrm{O}_2^+$",
        "co2": "$\\mathrm{CO}_2^+$", "co2+": "$\\mathrm{CO}_2^+$"
    }
    if sp not in label_map:
        raise ValueError(f"Unsupported species: {species}")
    label = label_map[sp]
    # Add label
    bbox_props = dict(boxstyle='round,pad=0.3',
                      facecolor='black', edgecolor='none',
                      alpha=0.6)
    ax.text(0.97, 0.97, label, transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)
    ax.text(0.97, 0.1, t_str, transform=ax.transAxes, color='white',
            ha='right', va='top', fontsize=12, bbox=bbox_props)
    return ax, cax

#%% Show the STATIC density data
def plot_d1_flux(ax,tint,yscale: str = "log",
                 markersize: float = 7.0,
                 linewidth: float = 1.0,
                 correct_bkg: bool = False,
                 ):
    """
    Plot MAVEN STATIC ion densities (H+, O+, O2+) over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis to plot on. If None, a new figure and axis are created.
    tint : list of str
        Time interval in ISO format, e.g., ["2023-09-17T05:27:00", "2023-09-17T05:35:00"].
    yscale : str, optional
        Y-axis scale, either "linear" or "log". Default is "linear".
    markersize : float, optional
        Size of markers for data points. Default is 7.0.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the density plot.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    tint_new = pyrf.extend_tint(tint, [-16.0, 16.0])
    d1 = maven.get_data(tint_new, "static_d1")

    if correct_bkg:
        moment_h = maven.static.moments_d1(d1, "H", Emin=0, Emax=30000,
                                           correct_background=True, correct_vsc=True)
        moment_o = maven.static.moments_d1(d1, "O", Emin=0, Emax=30000,
                                           correct_background=True, correct_vsc=True)
        moment_o2 = maven.static.moments_d1(d1, "O2", Emin=0, Emax=30000,
                                            correct_background=True, correct_vsc=True)
    else:
        moment_h = maven.static.moments_d1(d1, "H", Emin=0, Emax=30000,
                                           correct_background=False, correct_vsc=True)
        moment_o = maven.static.moments_d1(d1, "O", Emin=0, Emax=30000,
                                           correct_background=False, correct_vsc=True)
        moment_o2 = maven.static.moments_d1(d1, "O2", Emin=0, Emax=30000,
                                            correct_background=False, correct_vsc=True)

    Vt_h = norm(moment_h["V"].data).reshape(-1)
    Vt_o = norm(moment_o["V"].data).reshape(-1)
    Vt_o2 = norm(moment_o2["V"].data).reshape(-1)

    flux_h = np.abs(moment_h["n"].data * Vt_h * 1e5)
    flux_o = np.abs(moment_o["n"].data * Vt_o * 1e5)
    flux_o2 = np.abs(moment_o2["n"].data * Vt_o2 * 1e5)
    flux_h = ts_scalar(d1["time"], flux_h)
    flux_o = ts_scalar(d1["time"], flux_o)
    flux_o2 = ts_scalar(d1["time"], flux_o2)

    plot_line(ax, flux_h,  label="H",  color="black", linestyle="-", linewidth = linewidth,
                   marker=".", markersize=markersize)
    plot_line(ax, flux_o,  label="O",  color="blue",  linestyle="-", linewidth = linewidth,
                   marker=".", markersize=markersize)
    plot_line(ax, flux_o2, label="O2", color="red",   linestyle="-", linewidth = linewidth,
                   marker=".", markersize=markersize)
    ax.set_ylabel(r"$Flux\;(\mathrm{cm}^{-2}{s}^{-1})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    ax.set_yscale(yscale)
    ax.legend(
        [r"$\mathrm{H}^+$", r"$\mathrm{O}^+$", r"$\mathrm{O}_2^+$"],
        loc="center right",              # Align legend box to center right
        bbox_to_anchor=(1.1, 0.5),       # Place anchor point just outside the right axis
        frameon=False,
        fontsize=12,
        handlelength=0.5,
        ncol=1)
    return ax

#%% Show the SWEA omni data
def plot_swea_omni(ax, tint, cmap = 'Spectral_r', clim=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    tint_new = pyrf.extend_tint(tint, [-16.0, 16.0])
    swea = maven.get_data(tint_new, "swea_omni")

    if clim is None:
        clim = [1e4, 1e7]
    ax, cax = plot_spectr(ax, swea, yscale="log", cscale="log",
                          cmap=cmap, clim=clim)
    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.6, width=0.010)
    ax.set_ylabel("SWEA" + "\n" + "E [eV]", fontname='Times New Roman', fontsize=12)
    cax.set_ylabel("DEF\n[keV/(cm$^2$ s sr keV)]")
    ax.set_ylim([3, 3000])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax, cax


#%% Show the SWEA epad data
def plot_swea_pad(ax, tint, energyrange = [100.0, 1000.0],cmap = 'Spectral_r',
                  option="norm"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    swea_pad = maven.swea.get_pad(tint, delta_angles=22.5)
    # Merge over energy range to get PAD(time, pitchangle)
    pad_range = vdf.pitchangle_merge_energy(
        swea_pad,
        energyrange=energyrange,
        option = option
    )

    # Plot PAD
    if option == "norm":
        ax, cax = plot_spectr(ax, pad_range, cmap=cmap, clim=[0, 2])
        cax.set_ylabel("Norm(DEF)")
    else:
        ax, cax = plot_spectr(ax, pad_range, cscale="log", cmap=cmap)
        cax.set_ylabel("DEF\n[keV/(cm$^2$·s·sr·keV)]")

    plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.6, width=0.015)
    ax.set_ylabel("SWEA PAD [$^\\circ$]")
    ax.set_ylim([0, 180])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))

    e1, e2 = energyrange
    if e2 < 1000:
        label_str = f"{e1:.0f}–{e2:.0f} eV"
    else:
        label_str = f"{e1/1000:.1f}–{e2/1000:.1f} keV"
    plot.add_text(ax, label_str, 0.98, 0.98, color = 'white', facecolor = 'gray')
    return ax, cax
#%% Show the SWEA resample epad data
def plot_swea_resample_pad(ax, filename, tint, energyrange = [100.0, 1000.0],cmap = 'Spectral_r', option="norm"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    swea_pad = maven.swea.read_resample_pad(filename, tint)
    # Merge over energy range to get PAD(time, pitchangle)
    pad_range = vdf.pitchangle_merge_energy(
        swea_pad,
        energyrange=energyrange,
        option = option
    )

    # Plot PAD
    if option == "norm":
        ax, cax = plot_spectr(ax, pad_range, cmap=cmap, clim=[0.5, 1.5], shading='gouraud')
        cax.set_ylabel("Norm(DEF)")
    else:
        ax, cax = plot_spectr(ax, pad_range, cscale="log", cmap=cmap, shading='gouraud')
        cax.set_ylabel("DEF\n[keV/(cm$^2$·s·sr·keV)]")

    cax = plot.adjust_colorbar(ax, cax, 0.005, height_ratio=0.6, width=0.015)
    ax.set_ylabel("SWEA" +"\n"+ "PAD [$^\\circ$]")
    ax.set_ylim([0, 180])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))

    e1, e2 = energyrange
    if e2 < 1000:
        label_str = f"{e1:.0f}–{e2:.0f} eV"
    else:
        label_str = f"{e1/1000:.1f}–{e2/1000:.1f} keV"
    plot.add_text(ax, label_str, 0.98, 0.98, color = 'white', facecolor = 'black',
                  fontsize=13)
    return ax, cax

#%%
def plot_crustal_field_map(ax=None, option: str = "Bt"):
    """
    Plot crustal magnetic field map with a contour line at Bt = 50 nT.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot or None
        Axis to plot on. If None, a new figure and axis will be created.
    option : str
        Currently unused, placeholder for potential extensions.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    res = maven.get_lang_19_map()  # Should return a dict with 'lon', 'lat', and 'bt'

    # Plot filled contours (optional, for background)
    # cf = ax.contourf(res["lon"], res["lat"], res["bt"], levels=100, cmap="viridis")

    # Plot the Bt = 50 nT contour line
    cs = ax.contour(res["lon"], res["lat"], res["btot"], levels=[15, 30], colors='black', linewidths=0.5)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_xlim([00, 360])
    ax.set_ylim([-90, 90])
    return ax

if __name__ == "__main__":
    filename = r'D:\Work_Work\Mars\MAVEN\HCS_to_FB\20250615_113000_20250615_123000.cdf'
    # Define a time interval to test slicing
    tint = ["2025-06-15T11:30:00", "2025-06-15T12:30:00"]
    # plot_sta_c6(None, tint,['O+','O2+'])
    # plot_crustal_field_map(None)
    # plot_d1_flux(None, tint, yscale="log", markersize=8.0)
    plot_swea_resample_pad(None, filename, tint)
    plt.show()

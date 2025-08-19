"""
@author: Chi Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, LogNorm
from py_space_zc.plot import plot_line, plot_spectr, set_axis
import py_space_zc.maven as maven
from py_space_zc import pad_split_energy, ts_spectr
import matplotlib.cm as cm
from typing import Union, Iterable, Tuple, Optional

#%% Show the Bmso data (1Hz)
def plot_B(ax, tint):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    B = maven.get_data(tint, "B")
    plot_line(ax, B["Bmso"])
    ax.set_ylabel(r"$B_{\mathrm{MSO}}\ (\mathrm{nT})$")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax

#%% Show the Bmso data (32Hz)
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

#%% Show the Bmse data (1Hz)
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

#%% Show the SWIA omni data
def plot_swia_omni(ax, tint, clim=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    swia = maven.get_data(tint, "swia_omni")

    if clim is None:
        clim = [1e4, 1e7]
    ax, cax = plot_spectr(ax, swia["omni_flux"], yscale="log", cscale="log",
                          cmap="Spectral_r", clim=clim)

    ax.set_ylabel("$SWIA$\n$E_i$ [eV]")
    cax.set_ylabel("DEF\n[keV/(cm$^2$ s sr keV)]")
    ax.set_ylim([25, 20000])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax, cax

# %% Show the STATIC C6 omni data
def plot_sta_c6(ax, tint, species, clim=None, correct_background=False):
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
    c6_ref = maven.static.extract_data_c6(tint, sp_list[0], correct_background)
    time_ref = np.asarray(c6_ref.time.data)
    energy_ref = np.asarray(c6_ref.energy.data)
    DEF_sum = np.array(c6_ref.data, copy=True)  # avoid modifying the original

    # --- For any additional species, extract and sum on the same grid ---
    for sp in sp_list[1:]:
        c6_i = maven.static.extract_data_c6(tint, sp, correct_background)

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
        clim = [1e4, 5e7]

    ax, cax = plot_spectr(ax, c6_combined, yscale="log", cscale="log",
                          clim=clim, cmap='Spectral_r')

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
    ax.set_xlabel("Time")
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    cax.set_ylabel(r"DEF" "\n" r"[keV/(cm$^2$ s sr keV)]")

    # --- Corner label box ---
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='black',
                      edgecolor='none', alpha=0.6)
    ax.text(0.99, 0.95, label, transform=ax.transAxes, color='white',
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
    cscale: str = "log",
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
    vg_ms_for_reduce = None
    xylim_kms = (-500.0, 500.0)  # default display limits in km/s
    if vg_2d is not None:
        vg_arr = np.asarray(vg_2d, dtype=float)
        vg_ms_for_reduce = vg_arr * 1e3
        xylim_kms = (float(np.nanmin(vg_arr)), float(np.nanmax(vg_arr)))

    # 3) Perform reduction
    f2d = maven.static.reduced_d1_2d(
        d1, t_2d, mso_axis1, mso_axis2, species, vg_2d=vg_ms_for_reduce,
        correct_background = correct_background,
        correct_vsc = correct_vsc
    )

    # 4) Average over time dimension if present
    f2d_for_plot = f2d
    try:
        if hasattr(f2d, "dims") and ("time" in getattr(f2d, "dims", [])):
            f2d_for_plot = f2d.mean(dim="time")
        elif hasattr(f2d, "ndim") and f2d.ndim >= 3:
            f2d_for_plot = f2d.mean(axis=0)
    except Exception:
        pass

    # 5) Draw spectrogram
    ax, cax = plot_spectr(ax, f2d_for_plot, cscale=cscale, cmap=cmap, clim=clim)

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
    t_vals = np.array(f2d.time)
    if t_vals.size == 1:
        t_str = np.datetime_as_string(t_vals[0], unit='s')[11:]
    else:
        t_min_str = np.datetime_as_string(t_vals.min(), unit='s')[11:]
        t_max_str = np.datetime_as_string(t_vals.max(), unit='s')[11:]
        t_str = t_min_str

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



#%% Show the SWEA omni data
def plot_swea_omni(ax, tint, clim=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    swea = maven.get_data(tint, "swea_omni")

    if clim is None:
        clim = [1e4, 1e7]
    ax, cax = plot_spectr(ax, swea, yscale="log", cscale="log",
                          cmap="Spectral_r", clim=clim)

    ax.set_ylabel("$SWEA$\n$E_e$ [eV]")
    cax.set_ylabel("DEF\n[keV/(cm$^2$ s sr keV)]")
    ax.set_ylim([3, 3000])
    ax.set_xlim(np.datetime64(tint[0]), np.datetime64(tint[1]))
    return ax, cax


#%% Show the SWEA epad data
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
    tint = ["2015-09-21T13:00", "2015-09-21T16:00"]
    res = maven.get_lang_19_map()  # Should return a dict with 'lon', 'lat', and 'bt'
    # plot_sta_c6(None, tint,['O+','O2+'])
    plot_crustal_field_map(None)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from py_space_zc import maven, plot


DEFAULT_PANELS = ["sta_H", "sta_O", "sta_O2", "swia_omni", "swea_omni", "B"]

PANEL_ALIASES = {
    "b": "B",
    "mag": "B",
    "magnetic": "B",
    "b_high": "B_high",
    "bhigh": "B_high",
    "b_mse": "B_mse",
    "bmse": "B_mse",
    "swia": "swia_omni",
    "swia_omni": "swia_omni",
    "swia_energy": "swia_omni",
    "swia_density": "swia_density",
    "swia_n": "swia_density",
    "swia_velocity": "swia_velocity",
    "swia_v": "swia_velocity",
    "swia_pad": "swia_pad",
    "swea": "swea_omni",
    "swea_omni": "swea_omni",
    "swea_energy": "swea_omni",
    "swea_pad": "swea_pad",
    "h": "sta_H",
    "h+": "sta_H",
    "sta_h": "sta_H",
    "static_h": "sta_H",
    "sta_h+": "sta_H",
    "o": "sta_O",
    "o+": "sta_O",
    "sta_o": "sta_O",
    "static_o": "sta_O",
    "sta_o+": "sta_O",
    "o2": "sta_O2",
    "o2+": "sta_O2",
    "sta_o2": "sta_O2",
    "static_o2": "sta_O2",
    "sta_o2+": "sta_O2",
    "sta_c8": "sta_c8",
    "static_c8": "sta_c8",
    "sta_density": "sta_density",
    "sta_dens": "sta_density",
    "static_density": "sta_density",
    "d1_flux": "d1_flux",
}


def _plot_B(ax, tint):
    return maven.plot_B(ax, tint, legend_fontsize=12, ylabel_fontsize=12)


def _plot_B_high(ax, tint):
    return maven.plot_B_high(ax, tint, legend_fontsize=12, ylabel_fontsize=12)


def _plot_B_mse(ax, tint):
    return maven.plot_B_mse(ax, tint)


def _plot_swia_omni(ax, tint):
    ax, cax = maven.plot_swia_omni(ax, tint)
    cax.set_ylabel("")
    return ax


def _plot_swea_omni(ax, tint):
    ax, cax = maven.plot_swea_omni(ax, tint, clim=[1e5, 5e8])
    cax.set_ylabel("")
    return ax


def _plot_swia_density(ax, tint):
    return maven.plot_swia_density(ax, tint, ylabel_fontsize=12)


def _plot_swia_velocity(ax, tint):
    return maven.plot_swia_velocity(ax, tint, legend_fontsize=11, ylabel_fontsize=12)


def _plot_swia_pad(ax, tint):
    ax, cax = maven.plot_swia_pad(ax, tint)
    cax.set_ylabel("")
    return ax


def _plot_swea_pad(ax, tint):
    ax, cax = maven.plot_swea_pad(ax, tint)
    cax.set_ylabel("")
    return ax


def _plot_sta_c6_species(ax, tint, species):
    ax, cax = maven.plot_sta_c6(ax, tint, species, correct_background=False)
    ax.set_ylim(5, 30000)
    cax.set_ylabel("")
    return ax


def _plot_sta_H(ax, tint):
    return _plot_sta_c6_species(ax, tint, "H")


def _plot_sta_O(ax, tint):
    return _plot_sta_c6_species(ax, tint, "O")


def _plot_sta_O2(ax, tint):
    return _plot_sta_c6_species(ax, tint, "O2")


def _plot_sta_c8(ax, tint):
    return maven.plot_sta_c8(ax, tint)


def _plot_sta_density(ax, tint):
    return maven.plot_sta_dens(ax, tint)


def _plot_d1_flux(ax, tint):
    return maven.plot_d1_flux(ax, tint)


_PANEL_PLOTTERS = {
    "B": _plot_B,
    "B_high": _plot_B_high,
    "B_mse": _plot_B_mse,
    "swia_omni": _plot_swia_omni,
    "swia_density": _plot_swia_density,
    "swia_velocity": _plot_swia_velocity,
    "swia_pad": _plot_swia_pad,
    "swea_omni": _plot_swea_omni,
    "swea_pad": _plot_swea_pad,
    "sta_H": _plot_sta_H,
    "sta_O": _plot_sta_O,
    "sta_O2": _plot_sta_O2,
    "sta_c8": _plot_sta_c8,
    "sta_density": _plot_sta_density,
    "d1_flux": _plot_d1_flux,
}


def _apply_times_font(fig, fontname="Times New Roman"):
    """Force all axes, titles, labels, ticks, colorbars, and legends to use one font."""
    for ax in fig.axes:
        ax.title.set_fontname(fontname)
        ax.xaxis.label.set_fontname(fontname)
        ax.yaxis.label.set_fontname(fontname)
        ax.xaxis.get_offset_text().set_fontname(fontname)
        ax.yaxis.get_offset_text().set_fontname(fontname)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(fontname)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontname(fontname)


def _normalize_panels(panels):
    if panels is None:
        return DEFAULT_PANELS.copy()
    if isinstance(panels, str):
        panels = [panels]

    normalized = []
    unknown = []
    for item in panels:
        key = str(item).strip()
        canonical = PANEL_ALIASES.get(key.lower(), key)
        if canonical not in _PANEL_PLOTTERS:
            unknown.append(key)
        else:
            normalized.append(canonical)

    if unknown:
        valid = sorted(set(PANEL_ALIASES) | set(_PANEL_PLOTTERS))
        raise ValueError(f"Unknown panel(s): {unknown}. Valid panels include: {valid}")
    return normalized


def _plot_trajectory(ax_left, tint):
    B = maven.get_data(tint, "B")
    Pmvn = B["Pmso"].data / 3390.0
    Rmvn = np.sqrt(Pmvn[:, 1] ** 2 + Pmvn[:, 2] ** 2)
    t_mvn = B["Pmso"].time.data

    _, _, c1 = plot.scatter_time(
        ax_left[0], Pmvn[:, 0], Rmvn, t_mvn,
        cmap="Spectral_r", size=5.0, min_nticks=3)
    maven.bs_mpb(ax_left[0])
    plot.set_axis(
        ax_left[0], xlim=(-4.0, 2.0), ylim=(0.0, 5.5),
        tick_fontsize=12, label_fontsize=14,
        xlabel=r"$X_{\mathrm{MSO}}\ (R_{\rm M})$",
        ylabel=r"$\sqrt{Y_{\mathrm{MSO}}^2 + Z_{\mathrm{MSO}}^2}$ (R$_\mathrm{M}$)",
        facecolor="white", grid=False)
    ax_left[0].set_aspect("equal")
    c1.ax.set_ylabel("")
    plot.adjust_colorbar(ax_left[0], c1, 0.005, 0.8, 0.015)

    _, _, c2 = plot.scatter_time(
        ax_left[1], Pmvn[:, 1], Pmvn[:, 2], t_mvn,
        cmap="Spectral_r", size=5.0, min_nticks=3, zorder=20)
    maven.plot_mars(ax_left[1], texture=True, alpha=1.0, zorder=5)
    plot.set_axis(
        ax_left[1], xlim=(-4.0, 4.0), ylim=(-4.0, 4.0),
        tick_fontsize=12, label_fontsize=14,
        xlabel=r"$Y_{\mathrm{MSO}}\ (R_{\rm M})$",
        ylabel=r"$Z_{\mathrm{MSO}}\ (R_{\rm M})$",
        facecolor="white", grid=False)
    ax_left[1].set_aspect("equal")
    c2.ax.set_ylabel("")
    plot.adjust_colorbar(ax_left[1], c2, 0.005, 0.8, 0.015)


def show_overview(tint, panels=None):
    """
    Overview plot of MAVEN measurements over a given time interval.

    Parameters
    ----------
    tint : list[str]
        Time interval as [start_time, end_time], in ISO format.
    panels : list[str] or str, optional
        Right-column panels to draw. If omitted, the historical default is used:
        ["sta_H", "sta_O", "sta_O2", "swia_omni", "swea_omni", "B"].

        Common options:
        "B", "B_high", "B_mse", "swia_omni", "swia_density",
        "swia_velocity", "swia_pad", "swea_omni", "swea_pad",
        "sta_H", "sta_O", "sta_O2", "sta_c8", "sta_density", "d1_flux".

    Examples
    --------
    >>> maven.show_overview(tint, ["B", "swia_omni"])
    >>> maven.show_overview(tint, ["B", "swia_omni", "swea_omni", "swia_density"])
    """
    panel_names = _normalize_panels(panels)
    n_panels = len(panel_names)
    fig_height = max(6.0, 1.25 * n_panels + 1.2)
    fig = plt.figure(figsize=(13, fig_height))

    _, ax_left_grid = plot.subplot(
        2, 1, fig=fig, hspace=0.5, wspace=0.55,
        bottom=0.1, top=0.93, left=0.07, right=0.3)
    ax_left = np.asarray(ax_left_grid).reshape(-1)
    _plot_trajectory(ax_left, tint)

    _, ax_right_grid = plot.subplot(
        n_panels, 1, fig=fig, hspace=0.04, wspace=0.05,
        bottom=0.05, top=0.97, left=0.5, right=0.93, sharex=True)
    ax_right = np.asarray(ax_right_grid).reshape(-1)

    for i, (name, ax) in enumerate(zip(panel_names, ax_right)):
        _PANEL_PLOTTERS[name](ax, tint)
        if i == 0:
            plot.add_time_title(ax, tint, "yyyy/mm/dd HH:MM - HH:MM")
        if i < n_panels - 1:
            ax.set_xlabel("")
        plot.set_axis(ax, fontsize=12, tick_fontsize=12, label_fontsize=13, grid=False)

    _apply_times_font(fig)

    return fig

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_energy_pad_dot(
    ax,
    energy,
    pitchangle,
    psd,
    energyrange=None,
    cmap="jet",
    markersize=5,
    linewidth=1.5,
    marker="o",
    linestyle="-",
    yscale="log",
    alpha=1.0,
    label_fmt="{:.0f} eV",
    color_by="index",
    step=1,
    show_legend=True,
    legend_kwargs=None,
    colorbar=False,
    colorbar_label="Energy (eV)",
    xlabel="Pitch angle (deg)",
    ylabel="PSD",
    title=None,
    **kwargs,
):
    """Plot PAD curves at different energies with colors from a colormap.

    This is useful for Fu-style energy-resolved pitch-angle distribution plots:
    x is pitch angle, y is PSD or DEF, and each selected energy channel is one
    colored dot-line curve.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axis. If None, a new figure and axis are created.
    energy : array_like
        One-dimensional energy grid in eV.
    pitchangle : array_like
        One-dimensional pitch-angle grid in degrees.
    psd : array_like
        Two-dimensional PSD or DEF, shape (len(energy), len(pitchangle)).
    energyrange : sequence, optional
        Energy range to plot, in eV. If None, all energy channels are plotted.
    cmap : str or Colormap, optional
        Colormap used to assign colors by energy.
    markersize : float, optional
        Marker size in points.
    linewidth : float, optional
        Line width.
    marker, linestyle : str, optional
        Matplotlib marker and line style.
    yscale : str, optional
        Y-axis scale. Usually "log" for PSD or DEF.
    alpha : float, optional
        Marker and line transparency.
    label_fmt : str or None, optional
        Format string for legend labels. Set to None to suppress labels.
    color_by : {"index", "energy"}, optional
        If "index", selected energy channels are evenly spaced across the
        colormap. If "energy", colors are normalized by energy value.
    step : int, optional
        Plot every ``step`` energy channel after applying ``energyrange``.
        The default is 1, meaning all selected channels are plotted.
    show_legend : bool, optional
        Show legend when labels are available.
    legend_kwargs : dict, optional
        Extra keyword arguments passed to ``Axes.legend``.
    colorbar : bool, optional
        Add a colorbar showing the energy-color mapping.
    colorbar_label : str, optional
        Colorbar label.
    xlabel, ylabel, title : str, optional
        Axis labels and title.
    **kwargs
        Extra keyword arguments passed to ``Axes.plot``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with curves.
    lines : list
        Matplotlib line artists.
    cbar : matplotlib.colorbar.Colorbar or None
        Colorbar object if requested.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    elif not isinstance(ax, mpl.axes.Axes):
        raise TypeError("ax must be a matplotlib axes object or None.")

    energy = np.asarray(energy, dtype=float).ravel()
    pitchangle = np.asarray(pitchangle, dtype=float).ravel()
    data = np.asarray(psd, dtype=float)

    expected = (energy.size, pitchangle.size)
    if data.shape != expected:
        if data.shape == (pitchangle.size, energy.size):
            data = data.T
        else:
            raise ValueError(
                "psd must have shape (len(energy), len(pitchangle)) "
                f"or its transpose. Expected {expected}, got {data.shape}."
            )

    finite_energy = np.isfinite(energy)
    if energyrange is not None:
        erange = np.asarray(energyrange, dtype=float).ravel()
        if erange.size != 2:
            raise ValueError("energyrange must be None or contain two limits.")
        emin, emax = np.nanmin(erange), np.nanmax(erange)
        use = finite_energy & (energy >= emin) & (energy <= emax)
    else:
        use = finite_energy

    indices = np.where(use)[0]
    if indices.size == 0:
        raise ValueError("No energy channels are inside energyrange.")
    step = int(step)
    if step < 1:
        raise ValueError("step must be a positive integer.")
    indices = indices[::step]

    cmap_obj = plt.get_cmap(cmap)
    selected_energy = energy[indices]
    if color_by not in ("index", "energy"):
        raise ValueError("color_by must be 'index' or 'energy'.")

    if color_by == "energy" and np.nanmax(selected_energy) > np.nanmin(selected_energy):
        norm = mpl.colors.Normalize(float(np.nanmin(selected_energy)), float(np.nanmax(selected_energy)))
        color_positions = norm(selected_energy)
    elif color_by == "energy":
        norm = mpl.colors.Normalize(float(selected_energy[0] * 0.9), float(selected_energy[0] * 1.1))
        color_positions = norm(selected_energy)
    else:
        norm = mpl.colors.Normalize(0, max(indices.size - 1, 1))
        color_positions = np.linspace(0.0, 1.0, indices.size) if indices.size > 1 else np.array([0.5])

    lines = []
    for j, i in enumerate(indices):
        y = data[i, :]
        good = np.isfinite(pitchangle) & np.isfinite(y)
        if yscale == "log":
            good &= y > 0
        if not np.any(good):
            continue

        color = cmap_obj(color_positions[j])
        label = None if (label_fmt is None or not show_legend) else label_fmt.format(energy[i])
        line = ax.plot(
            pitchangle[good],
            y[good],
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            label=label,
            **kwargs,
        )[0]
        lines.append(line)

    cbar = None
    if colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(colorbar_label)
        if color_by == "index":
            cbar.set_ticks(np.arange(indices.size))
            cbar.set_ticklabels([label_fmt.format(e) if label_fmt is not None else f"{e:g}" for e in selected_energy])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_xlim(np.nanmin(pitchangle), np.nanmax(pitchangle))
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle="-", linewidth=0.2, color="0.6")
    ax.tick_params(axis="both", direction="in")

    if show_legend and label_fmt is not None and lines:
        if legend_kwargs is None:
            legend_kwargs = {}
        legend_defaults = {"fontsize": 8, "ncol": 2}
        legend_defaults.update(legend_kwargs)
        ax.legend(**legend_defaults)

    return ax, lines, cbar


__all__ = ["plot_energy_pad_dot"]

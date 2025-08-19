"""
Author: Chi Zhang
Date: August 2025

Description:
------------
Convenience function set_axis to streamline axis customization in matplotlib,
mimicking MATLAB's set(gca, ...) style, with extras:
- optional log axes (x/y)
- title + title fontsize
- spine (axis line) color/width
- optional tick color/width

Design:
-------
Only apply changes for arguments that are explicitly provided (not None).
Thus, existing axis settings are preserved unless you opt in.

Usage:
------
fig, ax = plt.subplots()
ax.set_yscale('log')  # pre-configured
set_axis(ax, xlim=(1, 100), xlabel='X')  # yscale remains 'log'
set_axis(ax, yscale='linear')            # now override to linear
"""

from typing import Iterable, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt

Number = Union[int, float]


def set_axis(
    ax,
    xlim: Optional[Tuple[Number, Number]] = None,
    ylim: Optional[Tuple[Number, Number]] = None,
    xticks: Optional[Sequence[Number]] = None,
    yticks: Optional[Sequence[Number]] = None,
    xticklabels: Optional[Sequence[str]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    fontsize: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
    label_fontsize: Optional[int] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    facecolor: Optional[Union[str, Tuple[float, float, float]]] = None,
    grid: Optional[bool] = None,  # None: keep, True/False: set
    legend: Optional[Union[bool, Sequence[str], str]] = None,  # None: keep; True: show; False: hide
    show_xticklabels: Optional[bool] = None,  # None: keep; True/False: set
    show_yticklabels: Optional[bool] = None,  # None: keep; True/False: set
    # --- extras ---
    xscale: Optional[str] = None,      # None: keep; "linear"/"log": set
    yscale: Optional[str] = None,      # None: keep; "linear"/"log": set
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    spine_color: Optional[str] = None,
    spine_width: Optional[float] = None,
    which_spines: Iterable[str] = ("left", "bottom", "right", "top"),
    tick_color: Optional[str] = None,
    tick_width: Optional[float] = None,
):
    """
    Set common axis properties in a concise way, similar to MATLAB's set(gca, ...) style.

    Only parameters explicitly provided (not None) will modify the axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to configure.
    xlim, ylim : (float, float), optional
        Axis limits.
    xticks, yticks : list of float, optional
        Tick positions.
    xticklabels, yticklabels : list of str, optional
        Custom tick labels (match lengths of ticks if provided).
    fontsize : int, optional
        Base font size for tick and axis labels.
    tick_fontsize : int, optional
        Tick label font size (overrides `fontsize` for ticks).
    label_fontsize : int, optional
        Axis label font size (overrides `fontsize` for labels).
    xlabel, ylabel : str, optional
        Axis labels.
    facecolor : str or RGB tuple, optional
        Axes background color.
    grid : bool, optional
        True to show grid, False to hide, None to keep current.
    legend : bool | list[str] | str | None
        True to show legend with existing labels,
        list/str to set labels and show legend,
        False to remove legend,
        None to keep current.
    show_xticklabels, show_yticklabels : bool, optional
        True/False to toggle visibility; None to keep current.
    xscale, yscale : {"linear","log"} or None
        Set axis scaling; None keeps current.
    title : str, optional
        Axes title (sets/overwrites only if provided).
    title_fontsize : int, optional
        Title font size (applied only if `title` is provided).
    spine_color : str, optional
        Color for selected spines (axis lines).
    spine_width : float, optional
        Linewidth for selected spines.
    which_spines : iterable of {"left","right","top","bottom"}, default all
        Which spines to style.
    tick_color : str, optional
        Color for tick marks and tick labels.
    tick_width : float, optional
        Line width for tick marks.
    """

    # Axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Tick positions
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Tick labels
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Font sizes
    if fontsize is not None:
        ax.tick_params(labelsize=fontsize)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
    if tick_fontsize is not None:
        ax.tick_params(labelsize=tick_fontsize)
    if label_fontsize is not None:
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)

    # Background
    if facecolor is not None:
        ax.set_facecolor(facecolor)

    # Grid
    if grid is not None:
        ax.grid(grid)

    # Tick label visibility
    if show_xticklabels is not None or show_yticklabels is not None:
        ax.tick_params(
            labelbottom=show_xticklabels if show_xticklabels is not None else ax.xaxis.get_ticklabels()[0].get_visible() if ax.xaxis.get_ticklabels() else True,
            labelleft=show_yticklabels if show_yticklabels is not None else ax.yaxis.get_ticklabels()[0].get_visible() if ax.yaxis.get_ticklabels() else True,
        )

    # Scales
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    # Title
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    # Spines (axis lines)
    if spine_color is not None or spine_width is not None:
        for sp in which_spines:
            spine = ax.spines.get(sp)
            if spine is None:
                continue
            if spine_color is not None:
                spine.set_color(spine_color)
            if spine_width is not None:
                spine.set_linewidth(spine_width)

    # Tick mark/label styling
    if tick_color is not None or tick_width is not None:
        ax.tick_params(
            axis='both',
            colors=tick_color if tick_color is not None else None,
            width=tick_width if tick_width is not None else None,
        )

    # Legend
    if legend is not None:
        if legend is False:
            lg = ax.get_legend()
            if lg is not None:
                lg.remove()
        elif isinstance(legend, (list, str)):
            ax.legend(legend)
        elif legend is True:
            ax.legend()


# Example (optional)
if __name__ == '__main__':
    import numpy as np
    x = np.logspace(0, 2, 100)
    y = x**0.5

    fig, ax = plt.subplots()
    ax.plot(x, y, label='y = x^0.5')
    ax.set_yscale('log')  # pre-configured

    # This will NOT change yscale because yscale=None by default
    set_axis(
        ax,
        xlim=(1, 100),
        ylim=(1, 10),
        xscale='log',
        xlabel='Frequency (Hz)',
        ylabel='Amplitude',
        title='Preserve-By-Default Example',
        title_fontsize=14,
        spine_color='k',
        spine_width=1.5,
        tick_color='k',
        tick_width=1.2,
        grid=None,        # keep grid as-is
        legend=True
    )

    plt.show()

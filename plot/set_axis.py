"""
Author: Chi Zhang
Date: August 2025

Description:
------------
This module defines a convenience function `set_axis` to streamline
axis customization in matplotlib, mimicking MATLAB's `set(gca, ...)` style.

It allows users to quickly set axis limits, tick positions and labels,
font sizes, axis labels, and background color using a clean and compact syntax.

Usage:
------
import matplotlib.pyplot as plt
from your_module import set_axis

fig, ax = plt.subplots()
ax.plot([...])
set_axis(ax, xlim=(...), xlabel='...', ...)

This function is especially useful for users transitioning from MATLAB
to Python or for repetitive figure formatting tasks.
"""

import matplotlib.pyplot as plt

def set_axis(
        ax,
        xlim=None,
        ylim=None,
        xticks=None,
        yticks=None,
        xticklabels=None,
        yticklabels=None,
        fontsize=None,
        tick_fontsize=None,
        label_fontsize=None,
        xlabel=None,
        ylabel=None,
        facecolor=None,
):
    """
    Set common axis properties in a concise way, similar to MATLAB's set(gca, ...) style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to configure.

    xlim, ylim : tuple of float, optional
        Axis limits, e.g., xlim=(-1, 1), ylim=(0, 5).

    xticks, yticks : list of float, optional
        Positions of ticks on x and y axes.

    xticklabels, yticklabels : list of str, optional
        Custom tick labels for x and y axes.

    fontsize : int, optional
        Font size for both tick labels and axis labels (overridden if tick_fontsize or label_fontsize is provided).

    tick_fontsize : int, optional
        Font size for tick labels only.

    label_fontsize : int, optional
        Font size for axis labels (xlabel and ylabel) only.

    xlabel, ylabel : str, optional
        Text labels for the x and y axes.

    facecolor : str or RGB tuple, optional
        Background color of the axis (e.g., 'white', '#f0f0f0', or (1.0, 1.0, 1.0)).
    """

    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set tick positions
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Set custom tick labels
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # Set axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Set overall font size (tick + axis labels)
    if fontsize is not None:
        ax.tick_params(labelsize=fontsize)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)

    # Override tick label font size if provided
    if tick_fontsize is not None:
        ax.tick_params(labelsize=tick_fontsize)

    # Override axis label font size if provided
    if label_fontsize is not None:
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)

    # Set axis background color
    if facecolor is not None:
        ax.set_facecolor(facecolor)


# Example usage
if __name__ == '__main__':
    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [2, 3, 4])

    # Apply axis formatting using set_axis
    set_axis(
        ax,
        xlim=(-1, 3),
        ylim=(1, 5),
        xticks=[0, 1, 2],
        yticks=[2, 3, 4],
        yticklabels=['low', 'mid', 'high'],
        xlabel='Time (s)',
        ylabel='Amplitude',
        tick_fontsize=12,
        label_fontsize=14,
    )

    # Display the plot
    plt.show()

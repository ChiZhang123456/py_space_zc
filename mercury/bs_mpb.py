# -*- coding: utf-8 -*-
"""
Created on 2025-07-30
Author: Chi Zhang

This module provides functions for plotting Mercury's bow shock and magnetopause
models based on empirical fits from Zhong et al. (2015) and Winslow et al. (2013).

Functions:
----------
- add_mp_zhong(ax):    Plot Zhong et al. (2015) magnetopause model.
- add_mp_winslow(ax):  Plot Winslow et al. (2013) magnetopause model.
- add_bs(ax):          Plot Winslow et al. (2013) bow shock model.
- bs_mpb(ax):          Quick visual overlay of BS + MP + Mercury disk.
"""

import numpy as np
import matplotlib.pyplot as plt


def add_mp_zhong(ax, **kwargs):
    """
    Plot Zhong et al. (2015) Mercury magnetopause model.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object on which to draw the curve.
    kwargs : dict
        Optional keyword arguments passed to ax.plot (e.g., color, linestyle).

    Model:
    ------
    R(θ) = Rmp * [2 / (1 + cosθ)]^power
    With Rmp = 1.51 Rm, power = 0.38
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    Rmp = 1.51
    power = 0.38
    R = Rmp * np.power(2. / (1 + np.cos(theta)), power)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    ax.plot(x, y, **kwargs)
    ax.set_xlim([-8,4])
    ax.set_ylim([-8,8])

def add_mp_winslow(ax, **kwargs):
    """
    Plot Winslow et al. (2013) Mercury magnetopause model.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object on which to draw the curve.
    kwargs : dict
        Optional keyword arguments passed to ax.plot.

    Model:
    ------
    R(θ) = Rss * [2 / (1 + cosθ)]^α
    With Rss = 1.45 Rm, α = 0.5
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    Rss = 1.45
    alpha = 0.5
    R = Rss * np.power(2. / (1 + np.cos(theta)), alpha)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    ax.plot(x, y, **kwargs)
    ax.set_xlim([-8,4])
    ax.set_ylim([-8,8])

def add_bs(ax, **kwargs):
    """
    Plot Winslow et al. (2013) Mercury bow shock model.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object on which to draw the curve.
    kwargs : dict
        Optional keyword arguments passed to ax.plot.

    Model:
    ------
    Bow shock in polar form:
        r(θ) = p * e / (1 + e * cosθ)
    Translated to x = r cosθ + x0
    With e = 1.04, p = 2.75, x0 = 0.5 Rm
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    x0 = 0.5
    ecc = 1.04
    p = 2.75
    L = p * ecc / (1 + ecc * np.cos(theta))
    x = L * np.cos(theta) + x0
    y = L * np.sin(theta)
    ax.plot(x, y, **kwargs)
    ax.set_xlim([-8,4])
    ax.set_ylim([-8,8])


def bs_mpb(ax):
    """
    Quick visualization of Mercury's bow shock, magnetopause (Winslow),
    and planetary disk.

    This function combines `add_bs()` and `add_mp_winslow()` into one plot,
    and draws a filled circle to represent Mercury.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object on which to draw.
    """


    add_mp_zhong(ax, color='k', linestyle='--', linewidth=1)
    add_bs(ax, color='k', linestyle='--', linewidth=1)

    # Add Mercury disk (two-tone)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 3600)
    ax.fill(np.cos(theta), np.sin(theta), '#D3D3D3', edgecolor='none')  # Dayside
    theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 3600)
    ax.fill(np.cos(theta), np.sin(theta), 'gray', edgecolor='none')  # Nightside

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlim([-8,4])
    ax.set_ylim([-8,8])

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # add_bs(ax)
    # add_mp_zhong(ax, color='b', linestyle='-', linewidth=0.5)
    # add_mp_winslow(ax, color = 'r', linestyle='-', linewidth=0.5)
    bs_mpb(ax)
    ax.set_aspect("equal")
    plt.show()

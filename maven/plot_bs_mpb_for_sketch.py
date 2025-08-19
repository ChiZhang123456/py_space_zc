# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:19:33 2025

@author: Win
"""
from py_space_zc import maven, plot
import matplotlib.pyplot as plt
import numpy as np

th = np.linspace(0, np.pi, 201)

# === Bow Shock (BS) ===
x0_bs, L_bs, ecc_bs = 0.6, 2.081, 1.026
rbs = L_bs / (1 + ecc_bs * np.cos(th))
xbs = rbs * np.cos(th) + x0_bs
rhobs = rbs * np.sin(th)
mask_bs = xbs <= 1.63
xbs = xbs[mask_bs]
rhobs = rhobs[mask_bs]

# === MPB dayside ===
x0_day, ecc_day, L_day = 0.640, 0.770, 1.080
rmpb_day = L_day / (1 + ecc_day * np.cos(th))
xmpb_day = rmpb_day * np.cos(th) + x0_day
rhompb_day = rmpb_day * np.sin(th)
mask_day = xmpb_day > 0
xmpb0 = xmpb_day[mask_day]
rhompb0 = rhompb_day[mask_day]

# === MPB nightside ===
x0_night, ecc_night, L_night = 1.600, 1.009, 0.528
rmpb_night = L_night / (1 + ecc_night * np.cos(th))
xmpb1 = rmpb_night * np.cos(th) + x0_night
rhompb1 = rmpb_night * np.sin(th)
mask_night = xmpb1 < 0
xmpb1 = xmpb1[mask_night]
rhompb1 = rhompb1[mask_night]

# === Full MPB
xmpb = np.concatenate((xmpb0, xmpb1))
rhompb = np.concatenate((rhompb0, rhompb1))

# === Draw boundaries ===
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xbs, rhobs, 'w-', linewidth=0.6)
ax.plot(xbs, -rhobs, 'w-', linewidth=0.6)
ax.plot(xmpb, rhompb, 'w-', linewidth=0.6)
ax.plot(xmpb, -rhompb, 'w-', linewidth=0.6)

# === Draw Mars ===
theta = np.linspace(-np.pi / 2, np.pi / 2, 500)
ax.fill(np.cos(theta), np.sin(theta), color='white', edgecolor='none', zorder=1)
theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 500)
ax.fill(np.cos(theta), np.sin(theta), color='gray', edgecolor='none', zorder=1)

# === Format axes ===
ax.set_aspect('auto')
ax.grid(False)
ax.set_xlim(-4, 2)
ax.set_ylim(-4, 4)
plot.set_axis(ax,show_yticklabels = False,show_xticklabels = False,
              facecolor = 'black' )

plt.show()
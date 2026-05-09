# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:51:02 2024

Author: Chi Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from py_space_zc.maven import bs_mpb
from py_space_zc.maven import get_data

def show_path_xr(ax, Pmso, cmap = 'Spectral_r'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    
    time = Pmso.time.data
    data = Pmso.data
    X = data[:,0] / 3390
    R = np.sqrt(data[:,1]**2 + data[:,2]**2) / 3390
    
    # Convert datetime64 values to Matplotlib date numbers.
    time_in_mpl = mdates.date2num(time.astype('datetime64[s]').astype('O'))

    bs_mpb(ax)
    scatter = ax.scatter(X, R, c=time_in_mpl, s=15, cmap=cmap, edgecolors='None')
    
    ax.set_xlabel(r"$X_{\mathrm{MSO}}$ (Rm)")
    ax.set_ylabel(r"$\sqrt{Y_{\mathrm{MSO}}^2 + Z_{\mathrm{MSO}}^2}$ (Rm)")

    # Add a colorbar outside the axes.
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.04, pad=0.005, aspect=14)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    #cbar.set_label('Time', labelpad=-50, rotation=270)
    ax.set_facecolor('white')
    return ax, scatter, cbar




# Example usage.
# Assume Pmso is a properly configured xarray.DataArray or similar time-series object.
# show_path_xr(None, Pmso)

 
if __name__ == "__main__":
    tint = ["2023-07-31T04:33:30", "2023-07-31T04:41:00"]
    B = get_data(tint, "B")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    show_path_xr(ax, B["Pmso"])
    plt.show()

    

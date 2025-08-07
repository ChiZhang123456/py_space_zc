# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:26:03 2024

@author: Win
"""

from py_space_zc import maven, emm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
from datetime import datetime


cmap = cm.Greens_r.copy()
cmap.set_bad(color='white')

#get the crustal fields map
cf_model=maven.get_lang_19_map();

def show_fuv_aurora(ax, time):
    filename = emm.get_data(time, "osr")
    emm_aurora1304 = emm.get_nightside_map(filename, wavelength_limits=[130.4-2.0, 130.4 + 2.0])
    midnight = maven.get_midnight(emm_aurora1304['time'])
    lon_night_center = np.nanmean(midnight['midnight_lon'])
    lat2_fig = np.nan_to_num(emm_aurora1304['lat'], nan=-9999)
    lon2_fig = np.nan_to_num(emm_aurora1304['lon'], nan=-9999)

    # 如果ax是None，创建一个新的fig和ax
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=ccrs.Orthographic(central_longitude=lon_night_center, central_latitude=0))

    #map of the brightness
    
    pc = ax.pcolormesh(lon2_fig, lat2_fig, emm_aurora1304['R'], 
                       transform=ccrs.PlateCarree(), cmap=cmap, 
                       rasterized=True)
    
    pc.set_clim(vmin=0, vmax=10)
    cbar = plt.colorbar(pc, ax=ax, orientation='vertical', 
                        shrink=0.5, pad=0.05, extend='max')
    cbar.ax.set_ylabel('OI 130.4 nm Brightness, R')
    ax.set_global()
    ax.gridlines(draw_labels=True,linewidth=1, color='gray', 
                 alpha=0.5, linestyle='--')
    ax.text(-0.08, 0.55, 'Latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=12)
    ax.text(0.5, -0.1, 'Longitude', va='bottom', ha='center',
        transform=ax.transAxes, fontsize=12)
    
    start_time = datetime.strptime(emm_aurora1304['time'][0], '%Y-%m-%dT%H:%M:%S.%f')
    end_time = datetime.strptime(emm_aurora1304['time'][-1], '%Y-%m-%dT%H:%M:%S.%f')

    title_fig = start_time.strftime('%Y-%m-%d %H:%M') + '-----' + end_time.strftime('%Y-%m-%d %H:%M')

    ax.set_title(title_fig,fontdict={'fontsize': 15})
    
# add the crustal fields
    ax.contour(cf_model['lon'], cf_model['lat'], cf_model['br'], 
                          levels=[-10, 10], linewidths=1, colors=['black', 'black'], extend='both', transform=ccrs.PlateCarree())

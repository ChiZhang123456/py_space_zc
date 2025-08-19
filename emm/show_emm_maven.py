# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 21:16:05 2024

@author: Win
"""
from py_space_zc import maven, emm
from pyrfu import pyrf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import spiceypy as spice
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime, timedelta
import py_space_zc
from pyrfu.plot import plot_line, plot_spectr, use_pyrfu_style
import matplotlib.dates as mdates
import scipy.io as sio
from matplotlib.ticker import FixedLocator
#maven.load_maven_spice()
#%%

def show_emm_maven(tint:str, mode:str):
    # initialize 
    use_pyrfu_style(usetex=True)
    cmap = cm.Greens_r.copy()
    cmap.set_bad(color='black')
    
    # load the crustal fields map
    cf_model = maven.get_lang_19_map();

#%% get the EMM data
    filename = emm.get_data(tint, mode)
    emm_aurora1304 = emm.get_nightside_map(filename,wavelength_limits=[130.4-2.0, 130.4 + 2.0])
    midnight = maven.get_midnight(emm_aurora1304['time'])
    lon_night_center = np.nanmean(midnight['midnight_lon'])

    #前后扩5分钟
    ts = py_space_zc.adjust_time(emm_aurora1304['time'][0], 0 * 60)
    te = py_space_zc.adjust_time(emm_aurora1304['time'][-1], 0 * 60)
    tint_mvn = [ts, te]

#%% get the MAVEN data
    B = maven.get_data(tint_mvn, "B") 
    lon, lat = maven.get_lon_lat(B["Pmso"])   # MAVEN longitude and latitude
    sza, alt = maven.get_sza_alt(B["Pmso"])   # MAVEN SZA and altitude
    Bmodel = maven.cf_model_mso(B["Pmso"])
    sza = pyrf.ts_scalar(B["Pmso"].time.data, sza, attrs = {"name":"solar zenith angle",
                                                            "unit":"degree"})

#%% plot the figure
    f = plt.figure(figsize=(14, 10))
#创建了一个主要的网格布局 gsp1，包括3行2列的子区域。
#此网格具有垂直间距 (hspace) 为0，意味着行之间没有间隙。bottom、top、left 和 right 参数控制整个网格相对于图形窗口边缘的位置，从而定义了图形的边距。
   
    gsp1 = f.add_gridspec(4, 2, hspace=0, bottom=0.09, top=0.95, left=0.1, right=0.9)
#从主网格 gsp1 中进一步细分出两个子网格布局。
#gsp10 被分为2行1列，位于主网格的第一行。gsp11 被分为3行1列，位于主网格的第二行。每个子网格的行间距也设置为0。
    gsp10 = gsp1[:, 0].subgridspec(2, 1, hspace=0.15)
    gsp11 = gsp1[:, 1].subgridspec(4, 1, hspace=0)

# Create axes in the grid spec
    axs10 = [f.add_subplot(gsp10[i]) for i in range(2)]
    axs11 = [f.add_subplot(gsp11[i]) for i in range(4)]

    f.subplots_adjust(wspace=0.35)
    maven.show_path_xr(axs10[0], B["Pmso"])
    try:
        maven.plot_B(axs11[0], tint_mvn)                         #maven magnetic field
    except:
        pass  # Ignore errors and continue  
    try:
        maven.plot_swia_omni(axs11[1], tint_mvn)                 #SWIA
    except:
        pass  # Ignore errors and continue  
    try:
        maven.plot_swea_omni(axs11[2], tint_mvn)                 #SWEA
    except:
        pass  # Ignore errors and continue  
    try:
        maven.plot_swea_epad(axs11[3], tint_mvn, "norm")         #SWEA electron pitch angle
    except:
        pass  # Ignore errors and continue  

    
    f.align_ylabels(axs11)

    # axs11[0].axvspan(
    #     mdates.datestr2num(emm_aurora1304['time'][0]),
    #     mdates.datestr2num(emm_aurora1304['time'][-1]),
    #     color="k",alpha=0.2)
    # axs11[1].axvspan(
    #     mdates.datestr2num(emm_aurora1304['time'][0]),
    #     mdates.datestr2num(emm_aurora1304['time'][-1]),
    #     color="k",alpha=0.2)
    # axs11[2].axvspan(
    #     mdates.datestr2num(emm_aurora1304['time'][0]),
    #     mdates.datestr2num(emm_aurora1304['time'][-1]),
    #     color="k",alpha=0.2)
    # axs11[3].axvspan(
    #     mdates.datestr2num(emm_aurora1304['time'][0]),
    #     mdates.datestr2num(emm_aurora1304['time'][-1]),
    #     color="k",alpha=0.2)
    
    for ax in axs11[:-1]:
        ax.set_xticklabels([])  # 清除x轴的刻度标签

#%% show EMM data
    original_pos = axs10[1].get_position()
    axs10[1].remove()
    axs_new = f.add_subplot(111, projection=ccrs.Orthographic(central_longitude=emm_aurora1304["lon_center"], 
                                                              central_latitude=emm_aurora1304["lat_center"]))
    axs_new.set_position(original_pos)  # 设置新Axes的位置为原来的位置
    lat2_fig = np.nan_to_num(emm_aurora1304['lat'], nan=-9999)
    lon2_fig = np.nan_to_num(emm_aurora1304['lon'], nan=-9999)
    pc = axs_new.pcolormesh(lon2_fig, lat2_fig, emm_aurora1304['R'], 
                       transform = ccrs.PlateCarree(), cmap=cmap, 
                       rasterized=True)
    
    pc.set_clim(vmin=0, vmax=10)
    axs_new.set_facecolor('black')

    axs_new.set_global()
    axs_new.gridlines(linewidth=1, color='gray', 
                 alpha=0.5, linestyle='--')
#写上纬度
    longitudes_tick = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    for lon_label in longitudes_tick:
        axs_new.text(lon_label, 0, f'{lon_label}°', transform=ccrs.Geodetic(), 
                 ha='center', va='bottom', fontsize=14, color='pink', weight='bold')

    axs_new.text(-0.08, 0.55, 'Latitude', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=axs_new.transAxes, fontsize=12)
    axs_new.text(0.5, -0.1, 'Longitude', va='bottom', ha='center',
        transform=axs_new.transAxes, fontsize=12)
    
    start_time = datetime.strptime(emm_aurora1304['time'][0], '%Y-%m-%dT%H:%M:%S.%f')
    end_time = datetime.strptime(emm_aurora1304['time'][-1], '%Y-%m-%dT%H:%M:%S.%f')

    title_fig = start_time.strftime('%Y-%m-%d %H:%M') + '-----' + end_time.strftime('%Y-%m-%d %H:%M')

    axs_new.set_title(title_fig,fontdict={'fontsize': 15})
    
# add the crustal fields
    axs_new.contour(cf_model['lon'], cf_model['lat'], cf_model['btot'], 
                          levels=[20], linewidths=1, colors=['cyan'], extend='both', transform=ccrs.PlateCarree())
    time_in_mpl = mdates.date2num(B["Pmso"].time.data.astype('O'))
    axs_new.scatter(lon, lat, c = time_in_mpl,s=15, cmap="Spectral_r", edgecolors='None',
                    transform=ccrs.PlateCarree())

    axs_new.set_position(original_pos)  # 设置新Axes的位置为原来的位置
    
#%% 
if __name__ == "__main__":    
    tint = "2023-11-26T13:07:00"
    maven.load_maven_spice()
    filename = emm.get_data(tint, "osr")
    emm_aurora1304 = emm.get_nightside_map(filename,wavelength_limits=[130.4-2.0, 130.4 + 2.0])
    show_emm_maven(tint, "osr")
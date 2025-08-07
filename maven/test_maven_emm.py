# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 08:53:34 2024

@author: Win
"""
from pyrfu import maven, pyrf, emm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import spiceypy as spice
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import util_zc
from pyrfu.plot import plot_line, plot_spectr, use_pyrfu_style
import matplotlib.dates as mdates
import scipy.io as sio
use_pyrfu_style(usetex=True)
#maven.load_maven_spice()
#%%
emm.download_data("2022-01-01","2023-12-31","os1")
emm.download_data("2022-01-01","2023-12-31","os2")

#%%
filename = "D:\\Work_Work\\Mars\\MAVEN\\sinuous_aurora\\sinuous_aurora_properties.mat"
mat = sio.loadmat(filename)
ts_aurora = util_zc.datenum2datetime64(mat["tint"][:,0])
te_aurora = util_zc.datenum2datetime64(mat["tint"][:,0])
max_sza = mat["max_sza"]
mask = max_sza >= 90
ts_aurora = ts_aurora[np.squeeze(mask)]
te_aurora = te_aurora[np.squeeze(mask)]

ts_aurora = ts_aurora[ts_aurora>= np.datetime64("2023-04-08T00:00")]
ts_aurora = pyrf.datetime642iso8601(ts_aurora)
#%%
path_out = "D:\\Work_Work\\Mars\MAVEN\\sinuous_aurora\\case\\"
for i, tint in enumerate(ts_aurora):
    # get the time interval of EMM
    filename = emm.get_data(tint, "osr")
    emm_aurora1304 = emm.get_nightside_map(filename,wavelength_limits=[130.4-2.0, 130.4 + 2.0])
    start_time = datetime.strptime(emm_aurora1304['time'][0], '%Y-%m-%dT%H:%M:%S.%f')
    end_time = datetime.strptime(emm_aurora1304['time'][-1], '%Y-%m-%dT%H:%M:%S.%f')
    # show the EMM - MAVEN conjunction
    emm.show_emm_maven(tint, "osr")
    name_out = path_out + f"Case_{i+1}" + '___'+ start_time.strftime('%Y%m%d_%H%M') + '___' + end_time.strftime('%Y%m%d_%H%M')
    plt.savefig(name_out, bbox_inches='tight', dpi=300, facecolor='w')
    plt.close()
    

#%%
spice.kclear()

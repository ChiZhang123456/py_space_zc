# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:05:32 2024

@author: Win
"""

import numpy as np
import xarray as xr

def pad_split_energy(pad, energyrange: list):
    energy = pad.energy.data
    coords = [pad.time.data, pad.theta.data]
    dims = ["time", "theta"]
    pad_data = pad.data.values
    
    if energy.ndim == 1:
       energy_indices = np.where((energy >= energyrange[0]) & (energy <= energyrange[1]))[0]
       pad_selected = pad_data[:, energy_indices, :]
       
    elif energy.ndim == 2:
       # 二维能量数组，假设第一维是时间，第二维是能量
       # 我们需要沿着每个时间点找到符合范围的能量索引
       energy_indices = np.where((energy >= energyrange[0]) & (energy <= energyrange[1]))
       pad_selected = np.array([pad_data[i, energy_indices[1][energy_indices[0] == i], :] for i in range(pad_data.shape[0])])

    else:
       raise ValueError("Energy array must be either 1D or 2D")       
          
       
    # 对选择的数据沿着能量维度求和
    pad_reduced = np.nanmean(pad_selected, axis=1)    

     
    res = xr.DataArray(
        pad_reduced,
        dims = dims, 
        coords = coords)
    

    return res


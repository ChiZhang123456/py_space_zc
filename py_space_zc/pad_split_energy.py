# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:05:32 2024

Author: Chi Zhang
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
       # For 2D energy arrays, assume dimensions are time and energy.
       # Select the energy indices within the requested range at each time.
       energy_indices = np.where((energy >= energyrange[0]) & (energy <= energyrange[1]))
       pad_selected = np.array([pad_data[i, energy_indices[1][energy_indices[0] == i], :] for i in range(pad_data.shape[0])])

    else:
       raise ValueError("Energy array must be either 1D or 2D")       
          
       
    # Average the selected data along the energy dimension.
    pad_reduced = np.nanmean(pad_selected, axis = 1)

     
    res = xr.DataArray(
        pad_reduced,
        dims = dims, 
        coords = coords)
    

    return res


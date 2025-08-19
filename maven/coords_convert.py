# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:11:37 2024

@author: Chi Zhang
"""

import spiceypy as spice
import numpy as np
from py_space_zc import irf_time, ts_vec_xyz

def coords_convert(inp, option):
    """
    MAVEN_COORDS - Transforms coordinates between different MAVEN-related frames.
    
    This function performs coordinate transformations for MAVEN data between
    various coordinate systems such as MAVEN_STATIC, MAVEN_MSO, IAU_MARS, etc.
    
    Parameters:
        Inp: Pyrf.time_vec_xyz data
        option (str): String specifying the coordinate transformation to be applied.
    Returns:
        Out: Pyrf.time_vec_xyz data
    """
    
    # Extract the time and data  
    time = inp.time.data
    data = inp.data
        
    # 将 datetime64 转换为 datetime
    time_dt = irf_time(time,"datetime64>datetime")
    
    # 将 datetime 转换为 ISO 格式的字符串
    iso_strings = [d.strftime("%Y-%m-%dT%H:%M:%S.%f") for d in time_dt]
    
    et = np.array([spice.str2et(iso_str) for iso_str in iso_strings])
    

    # Selecting the transformation matrix based on the option
    frame_from, frame_to, new_coords = {
        'sta2mso': ('MAVEN_STATIC', 'MAVEN_MSO', 'MSO'),
        'mso2sta': ('MAVEN_MSO', 'MAVEN_STATIC', 'MAVEN_STATIC'),
        'mso2pc': ('MAVEN_MSO', 'IAU_MARS', 'PC'),
        'pc2mso': ('IAU_MARS', 'MAVEN_MSO', 'MSO'),
        'mso2swea': ('MAVEN_MSO', 'MAVEN_SWEA', 'MAVEN_SWEA'),
        'swea2mso': ('MAVEN_SWEA', 'MAVEN_MSO', 'MSO'),
        'mso2swia': ('MAVEN_MSO', 'MAVEN_SWIA', 'MAVEN_SWIA'),
        'swia2mso': ('MAVEN_SWIA', 'MAVEN_MSO', 'MSO'),
        'pl2mso': ('MAVEN_APP_BP', 'MAVEN_MSO', 'MSO')
    }.get(option.lower(), (None, None, None))

    if frame_from is None or frame_to is None:
        raise ValueError('Invalid transformation option provided.')

    # Compute transformation matrices for all times
    matrices = np.array([spice.pxform(frame_from, frame_to, t) for t in et])
    
    # Apply the transformation matrix to each data point
    out = np.array([np.dot(matrices[i], data[i]) for i in range(len(data))])
    
    # Update the coordinates
    attrs_new = inp.attrs;
    attrs_new["coordinates"] = new_coords
    
    # Convert to pyrf.timeseries
    Out = ts_vec_xyz(inp.time.data, out, attrs_new)
    
    return Out

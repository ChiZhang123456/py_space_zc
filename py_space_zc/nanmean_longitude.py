# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:37:19 2024

Author: Chi Zhang
"""

import numpy as np

def nanmean_longitude(lon):
    """
    Compute the circular mean of longitude values in a 2D matrix.
    
    Parameters
    ----------
    lon : array-like
        2D longitude matrix in degrees.
    
    Returns
    -------
    circ_mean_lon : float
        Circular mean longitude in degrees.
    """
    # Convert longitude to radians.
    lon_rad = np.radians(lon)
    
    # Compute the mean cosine and sine components.
    cos_lon = np.nanmean(np.cos(lon_rad), axis=(0, 1))
    sin_lon = np.nanmean(np.sin(lon_rad), axis=(0, 1))
    
    # Compute the angle of the mean vector.
    circ_mean_lon_rad = np.arctan2(sin_lon, cos_lon)
    
    # Convert the result back to degrees.
    circ_mean_lon = np.degrees(circ_mean_lon_rad)
    
    return circ_mean_lon

if __name__ == "__main__":
    lon = np.array([[1, 359], [2, 358], [np.nan, 3]])  # Example data
    mean_longitude = nanmean_longitude(lon)
    print("Circular mean longitude:", mean_longitude)

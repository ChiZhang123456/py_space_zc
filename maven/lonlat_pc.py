# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:51:28 2024

@author: Win
"""

import numpy as np
import pyrfu.pyrf as pyrf

def lonlat2pc(alt, lon, lat):
    """
    Convert geographic coordinates (altitude, longitude, latitude) on Mars to Cartesian coordinates.
    
    Parameters:
    - alt (numpy.ndarray): Altitudes in km (1D array).
    - lon (numpy.ndarray): Longitudes in degrees (1D array).
    - lat (numpy.ndarray): Latitudes in degrees (1D array).
    
    Returns:
    - numpy.ndarray: Nx3 array of Cartesian coordinates [x, y, z].
    """
    # Mars radius in kilometers
    R = 3393.0

    # Convert degrees to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # Spherical to Cartesian conversion
    r = R + alt
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.column_stack((x, y, z))


def pc2lonlat(Pc):
    lon, lat, radius = pyrf.cart2sph(Pc[:,0], Pc[:,1], Pc[:,2])
    lon, lat = np.degrees(lon), np.degrees(lat)
    lon[lon <= 0] += 360 
    lat = 90 - lat
    return lon, lat

# Example usage:
# altitudes = np.array([0])  # km above Martian surface
# longitudes = np.array([0])  # degrees
# latitudes = np.array([90])  # degrees

# cartesian_coords = lonlat_to_pc(altitudes, longitudes, latitudes)
# print("Cartesian Coordinates:\n", cartesian_coords)

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:53:27 2024

@author: Chi Zhang
"""

import numpy as np
import spiceypy as spice
from pyrfu import pyrf
from py_space_zc import irf_time
from .coords_convert import coords_convert

#%%
def get_pos_spice(time):
    """
    time is str, like "2022-09-17T01:00"
    """
    et = spice.str2et(time)
    pos, _ = spice.spkpos('MAVEN', et, 'IAU_MARS', 'NONE', 'MARS')
    lon, lat, radius = pyrf.cart2sph(pos[:,0], pos[:,1], pos[:,2])
    lon, lat = np.degrees(lon), np.degrees(lat)
    lat[lat <= 0] += 360 
    pos, _ = spice.spkpos('MAVEN', et, 'MAVEN_MSO', 'NONE', 'MARS')
    Pmso = np.array([pos])
    res={'Pmso':Pmso,
         'lon':lon,
         'lat':lat}
    return res
#%%
def get_subsolar(time):
    """
    time is datetime64
    """  
    if isinstance (time, np.datetime64):
        time_dt = irf_time(time, 'datetime64>datetime')
    else:
        time_dt = [np.datetime64(t) for t in time]
        time_dt = irf_time(time_dt, 'datetime64>datetime')
    iso_strings = [d.strftime("%Y-%m-%dT%H:%M:%S.%f") for d in time_dt] 
    et = np.array([spice.str2et(iso_str) for iso_str in iso_strings])
    sun_pos_mars,_ = spice.spkpos('SUN', et, 'IAU_MARS', 'NONE', 'MARS')
    _, lon, lat = pyrf.cart2sph(sun_pos_mars[:,0], sun_pos_mars[:,1], sun_pos_mars[:,2])
    res={'subsolar_lon':lon,
         'subsolar_lat':lat}
    return res
#%%
def get_midnight(time):
    """
    Calculate the midnight point on Mars for a given time.
    This function uses the get_subsolar function and adjusts the longitude by 180 degrees.

    Parameters:
        time (np.datetime64): The time for which to calculate the midnight longitude and latitude.

    Returns:
        dict: A dictionary containing 'midnight_lon' and 'midnight_lat'.
    """
    # First, get the subsolar longitude and latitude using the get_subsolar function
    subsolar = get_subsolar(time)
    
    # Adjust the longitude by 180 degrees to get the midnight longitude
    # Ensure longitude stays within the range of 0 to 360 degrees
    midnight_lon = (subsolar['subsolar_lon'] + 180) % 360

    # The latitude for midnight is the same as the subsolar latitude
    midnight_lat = subsolar['subsolar_lat']
    
    # Return the results in a dictionary
    res = {
        'midnight_lon': midnight_lon,
        'midnight_lat': midnight_lat
    }
    return res

# Example usage
# time_example = np.datetime64('2023-01-01T12:00:00')
# print(get_midnight(time_example))
#%%
def get_lon_lat(Pmso):
    Ppc = coords_convert(Pmso, "mso2pc")
    lon, lat, radius = pyrf.cart2sph(Ppc.data[:,0], Ppc.data[:,1], Ppc.data[:,2])
    # lon : 0-2pi
    # lat: 0-pi, 从+Z轴旋转到-Z轴，但是实际上是 90，到-90
    lon, lat = np.degrees(lon), np.degrees(lat)
    lon[lon <= 0] += 360 
    lat = 90 - lat
    return lon, lat

#%%
def get_sza_alt(Pmso):
    """
    Computes the Solar Zenith Angle (SZA) and altitude.
    Input:
        Pmso - A matrix or DataArray where each row represents a point in 3D space (x, y, z).
    Output:
        sza - Solar Zenith Angle in degrees.
        alt - Altitude above the Martian surface in meters.
    Example:
        sza, alt = get_sza_alt(Pmso)
    """
    # Check if Pmso is a DataArray and extract the data if true
    if isinstance(Pmso, np.ndarray):
        data = Pmso
    else:
        try:
            data = Pmso.data  # Assuming DataArray from xarray
        except AttributeError:
            raise TypeError("Input must be either a numpy ndarray or a DataArray containing data.")
    
    if data.shape[1] != 3:
        raise ValueError("Input must be a Nx3 matrix or array.")

    # Calculate the distance from the origin
    distance = np.sqrt(np.sum(data**2, axis=1))

    # Solar Zenith Angle calculation
    sza = np.degrees(np.arctan(np.sqrt(data[:, 1]**2 + data[:, 2]**2) / data[:, 0]))
    sza[sza < 0] += 180

    # Altitude calculation
    alt = distance - 3393.19

    return sza, alt


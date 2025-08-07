# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:27:32 2024

@author: Chi Zhang
"""
from astropy.io import fits as pyfits
import numpy as np
import py_space_zc

def integrate_swath_radiance(integrate_values: np.ndarray, 
                             wavelength: np.ndarray,
                             wavelength_limits: list, 
                             return_radiance: bool = True) -> np.ndarray:
    lambda_min = wavelength_limits[0]
    lambda_max = wavelength_limits[1]

    mask = (lambda_min < wavelength) & (wavelength < lambda_max)
    mask = np.repeat(mask[np.newaxis, :, :], integrate_values.shape[0], axis=0)

    radiance = np.sum(np.where(mask, integrate_values, 0.), axis=2)

    if return_radiance:
        dlambda = \
            np.nanmean(np.where(mask[0, :, 1:],
                                np.diff(wavelength, axis=1),
                                np.nan))
        radiance *= dlambda

    return radiance


def subtract_baseline(wavelengths, radiance):

    baseline = np.zeros_like(radiance)
    corrected_radiance = np.zeros_like(radiance)

    num_integrations, num_spatial_bins, _ = radiance.shape

    for i in range(num_integrations):
        for j in range(num_spatial_bins):
            spectrum = radiance[i, j, :]

            left_bg_start = np.argmin(np.abs(wavelengths[j, :] - 127.2))
            left_bg_end = np.argmin(np.abs(wavelengths[j, :] - 128.2))
            emission_start = np.argmin(np.abs(wavelengths[j, :] - 128.4))
            emission_end = np.argmin(np.abs(wavelengths[j, :] - 132.4))
            right_bg_start = np.argmin(np.abs(wavelengths[j, :] - 132.6))
            right_bg_end = np.argmin(np.abs(wavelengths[j, :] - 133.6))

            left_region = range(left_bg_start, left_bg_end+1)
            right_region = range(right_bg_start, right_bg_end+1)
            baseline_indices = np.concatenate([left_region, right_region])

            fit = np.polyfit(wavelengths[j, baseline_indices], spectrum[baseline_indices], 1)
            fitted_baseline = np.polyval(fit, wavelengths[j, :])

            baseline[i, j, :] = fitted_baseline
            
            corrected_spectrum = spectrum - fitted_baseline

            corrected_radiance[i, j, :] = corrected_spectrum

    return baseline, corrected_radiance


def get_nightside_map(filename,wavelength_limits: list):
    # example:
        # filename = 'D:\Work_Work\Mars\EMM\kp_paper\emm_emu_l2a_20221201t002750_0301_osr_38566_f_v04-00.fits.gz' 
        # wavelength_limits = [130.4 - 2.0, 130.4 + 2.0]
        # read_emm.get_nightside_map
    
    # 使用pyfits打开FITS文件
    hdul = pyfits.open(filename)

    # 从FOV_GEOM扩展中读取纬度、经度、太阳天顶角和发射角
    lat = hdul['FOV_GEOM'].data['lat'][:,0,:]  # 纬度数据
    lon = hdul['FOV_GEOM'].data['lon'][:,0,:]  # 经度数据
    sza = hdul['FOV_GEOM'].data['SOLAR_Z_ANGLE'][:,0,:]  # 太阳天顶角
    ea = hdul['FOV_GEOM'].data['EMISSION_ANGLE'][:,0,:]  # 发射角
    
    # print(time.columns) 就知道里面的信息，分别是time_et和time_utc
    time = hdul['TIME'].data['TIME_UTC'] #时间
    Pmso=hdul['SC_GEOM'].data['V_SC_POS_MSO']
    
    # 读取辐射亮度和波长数据
    radiance = hdul['CAL'].data['RADIANCE']  # 辐射亮度
    wavelengths = hdul['WAVELENGTH'].data['WAVELENGTH_L2A'][0]  # 波长

    # 基线校正
    baseline, corrected_radiance = subtract_baseline(wavelengths, radiance)
    
    # 积分辐射亮度，限定在特定波长范围
    radiance = integrate_swath_radiance(corrected_radiance, wavelengths, wavelength_limits)

    # 筛选条件设置
    szalim = 90  # 太阳天顶角限制
    ealim = 90   # 发射角限制
    mask = ((sza > szalim) & (ea < ealim))  # 创建掩膜，筛选有效数据
    
    # 创建夜间照射区域的辐射亮度副本，并应用掩膜
    radiance_nightdisk = np.copy(radiance)
    radiance_nightdisk[~mask] = np.nan  # 非有效区域设为NaN
    lat[~mask] =np.nan
    lon[~mask] =np.nan
    
    # map center
    lat_center = np.nanmean(lat)
    lon_center = py_space_zc.nanmean_longitude(lon)
    
    # output:
    res={'time':time,
         'Pmso':Pmso,
         'lat':lat,
         'lon':lon,
         'lat_center':lat_center,
         'lon_center':lon_center,
         'R':radiance_nightdisk}
    
    return res  

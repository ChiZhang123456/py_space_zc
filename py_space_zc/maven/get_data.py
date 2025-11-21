# -*- coding: utf-8 -*-
"""
MAVEN Data Retrieval Module
Author: Chi Zhang (zc199508@bu.edu)
Date: Last updated in 2024
"""

import os
import numpy as np
import datetime
import glob
import scipy.io as sio
import h5py
import xarray as xr
from py_space_zc.maven import get_base_path, static, swia
from py_space_zc import irf_time, loadmat, get_cdf_var, read_time_from_file, year_month_day, tint_data, ts_scalar, ts_vec_xyz, ts_spectr, ts_skymap
from py_space_zc.vdf import create_pdist_skymap

#%%
def get_data(tint, var):
    """
    Retrieve MAVEN data for the specified time range and variable.

    Parameters
    ----------
    tint : list of str
        Time interval as a list of two ISO-format strings, e.g.,
        ['2018-11-01T01:00:00', '2018-11-01T02:00:00'].
    var : str
        Variable to retrieve. Supported options include:

        Magnetic field:
        - 'B'           : 1-second cadence magnetic field data (MSO coordinates, .mat files)
        - 'B_high'      : 32 Hz high-resolution magnetic field data (MSO coordinates, .mat files)
        - 'Bmse'        : 1-second magnetic field in MSE coordinates (.mat files)

        SWIA (Solar Wind Ion Analyzer):
        - 'swia_omni'   : Omnidirectional moments & DEF from SWIA onboard survey mode (.cdf)
        - 'swia_3d'     : 3D distribution function from coarse survey mode (.cdf)
        - 'swia_moment' : Derived ion moments from SWIA coarse 3d data  (.mat from Chi Zhang)

        SWEA (Solar Wind Electron Analyzer):
        - 'swea_omni'   : Omnidirectional electron spectra (.cdf)
        - 'swea_pad'    : Electron pitch angle distribution (.cdf)
        - 'swea_topo'   : SWEA top-hat field of view topography (.mat)
        - 'swea_scpot'  : SWEA-LPW spacecraft potential (.mat)

        STATIC (Suprathermal and Thermal Ion Composition):
        - 'static_c0'       : STATIC C0 mode (ion spectra, low energy range) (.cdf)
        - 'static_c6'       : STATIC C6 mode (wider mass range) (.cdf)
        - 'static_c8'       : STATIC C8 mode (wider direction range) (.cdf)
        - 'static_c6_iv4'   : STATIC C6 background counts (.cdf)
        - 'static_d1'       : STATIC D1 mode (ion VDF in 3D, high resolution) (.cdf)
        - 'static_d1_iv4'   : STATIC D1 background counts (.cdf)
        - 'static_density'  : Derived ion density from STATIC (Gwen's txt format)
        - 'static_moment'   : Derived ion moments from STATIC D1 mode (.mat from Chi Zhang)

    Returns
    -------
    dict or xarray.Dataset
        A dictionary or xarray object containing the retrieved data, depending on format.

    Example
    -------
    >>> from py_space_zc import maven
    >>> tint = ['2018-11-01T01:00:00', '2018-11-01T02:00:00']
    >>> B = maven.get_data(tint, 'B')               # 1s magnetic field
    >>> swea = maven.get_data(tint, 'swea_omni')    # Electron omnidirectional spectra
    >>> sta = maven.get_data(tint, 'static_d1')     # Ion VDF from STATIC D1

    Notes
    -----
    Internally, the function uses a configuration dictionary to determine file paths
    and filename formats for each variable. It supports multiple data formats, including
    MATLAB (.mat), CDF (.cdf), and ASCII (.txt).
    """

              
    # Define base path for MAVEN data
    base_path_mvn = get_base_path() 
    start_time, end_time = np.datetime64(tint[0]), np.datetime64(tint[1])

    # the path of the data
    data_config = {
        'B': {'path': os.path.join(base_path_mvn, 'mag', 'matlab_data_1s'),
              'filename_format': '{date}_1s.mat'},
        
        'B_high': {'path': os.path.join(base_path_mvn, 'mag', 'matlab_data'),
              'filename_format': '{date}_high.mat'},
        
        'Bmse': {'path': os.path.join(base_path_mvn, 'mag', 'MSE_new'),
            'filename_format': '????????_??????_????????_??????.mat'},
        
        'swia_omni': {'path': os.path.join(base_path_mvn, 'swi', 'l2'),
                      'filename_mom_format': 'mvn_swi_l2_onboardsvymom_{date}_*.cdf',
                      'filename_def_format': 'mvn_swi_l2_onboardsvyspec_{date}_*.cdf'},

        'swia_3d': {'path': os.path.join(base_path_mvn, 'swi', 'l2'),
                      'filename_format': 'mvn_swi_l2_coarsesvy3d_{date}_*.cdf',},

        'swea_omni':{'path': os.path.join(base_path_mvn, 'swe', 'l2'),
                      'filename_format': 'mvn_swe_l2_svyspec_{date}_*.cdf'},

        'swea_pad':{'path': os.path.join(base_path_mvn, 'swe', 'l2'),
                      'filename_format': 'mvn_swe_l2_svypad_{date}_*.cdf'},

        'swea_topo': {'path': os.path.join(base_path_mvn, 'swe', 'l3',),
                     'filename_format': 'topo_{date}.mat'},

        'swea_scpot': {'path': os.path.join(base_path_mvn, 'swe', 'l3','swea_lpw_pot_mat'),
                      'filename_format': 'swea_scpot_{date}.mat'},

        'static_c0':{'path': os.path.join(base_path_mvn, 'sta', 'l2'),
                      'filename_format': 'mvn_sta_l2_c0-64e2m_{date}_*.cdf'},

        'static_c6':{'path': os.path.join(base_path_mvn, 'sta', 'l2'),
                      'filename_format': 'mvn_sta_l2_c6-32e64m_{date}_*.cdf'},

        'static_c8': {'path': os.path.join(base_path_mvn, 'sta', 'l2'),
                      'filename_format': 'mvn_sta_l2_c8-32e16d_{date}_*.cdf'},

        'static_c6_iv4': {'path': os.path.join(base_path_mvn, 'sta', 'iv4'),
                      'filename_format': 'mvn_sta_l2_c6-32e64m_{date}_iv4.cdf'},

        'static_d1':{'path': os.path.join(base_path_mvn, 'sta', 'l2'),
                      'filename_format': 'mvn_sta_l2_d1-32e4d16a8m_{date}_*.cdf'},

        'static_d1_iv4': {'path': os.path.join(base_path_mvn, 'sta', 'iv4'),
                          'filename_format': 'mvn_sta_l2_d1-32e4d16a8m_{date}_iv4.cdf'},

        'static_density':{'path': os.path.join(base_path_mvn, 'sta', 'l3','density_Gwen_txt'),
                      'filename_format': 'den_{date}.txt'},

        'static_moment': {'path': os.path.join(base_path_mvn, 'sta', 'l3','moments_d1_ChiZhang'),
                           'filename_format': 'moments_d1_{date}.mat'},
    }
    
    config = data_config.get(var)
    
    if not config:
        raise ValueError(f"Unsupported variable: {var}")
    
    dates_to_read = np.arange(start_time.astype('datetime64[D]'), 
                              end_time.astype('datetime64[D]') + np.timedelta64(1, 'D'),
                              dtype='datetime64[D]')

    #%% read the magnetic field (1Hz), mso
    if var == 'B':    
        res = {'time': np.array([], dtype='datetime64[ns]'),
               'Bmso': np.array([], dtype=float).reshape(0, 3),
               'Pmso': np.array([], dtype=float).reshape(0, 3)}
        for date_file in dates_to_read:
            filename = os.path.join(config['path'], config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
            if os.path.exists(filename):
                mat_data = loadmat(filename)
                # convert the datenum (Matlab) to datetime (Python)
                time_array = np.array([datetime.datetime.fromordinal(int(t)) + 
                                   datetime.timedelta(days=t%1) - 
                                   datetime.timedelta(days=366) 
                                   for t in mat_data['time'].flatten()])
                time_array = irf_time(mat_data['time'],"datenum>datetime64")
                # select the data within the interval
                mask = (time_array >= np.datetime64(start_time)) & (time_array <= np.datetime64(end_time))
                res['time'] = np.concatenate((res['time'], time_array[mask]))
                res['Bmso'] = np.vstack((res['Bmso'], mat_data['Bmso'][np.squeeze(mask),:]))
                res['Pmso'] = np.vstack((res['Pmso'], mat_data['Pmso'][np.squeeze(mask),:]))
        
        # convert the data to TSeries
        B = ts_vec_xyz(res['time'], res['Bmso'], attrs={"name": "Magnetic field",
                                                                     "instrument": "MAG",
                                                                     "UNITS":"nT",
                                                                     "coordinates":"MSO",
                                                                     "resolution":"1Hz"})
        P = ts_vec_xyz(res['time'], res['Pmso'], attrs={"name": "Position",
                                                                     "UNITS":"km",
                                                                     "coordinates":"MSO",
                                                                     "resolution":"1Hz"})
        res = {'Bmso':B,
               'Pmso':P}
        return res

    #%% read the magnetic field (32Hz), mso
    elif var == 'B_high':
        res = {'time': np.array([], dtype='datetime64[ns]'),
               'Bmso': np.array([], dtype=np.float32).reshape(0, 3),
               'Pmso': np.array([], dtype=np.float32).reshape(0, 3)}
        for date_file in dates_to_read:
            filename = os.path.join(config['path'], config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
            if os.path.exists(filename):
                mat_data = loadmat(filename)
                time_array = irf_time(mat_data['time'],"datenum>datetime64")

                # select the data within the interval
                mask = (time_array >= np.datetime64(start_time)) & (time_array <= np.datetime64(end_time))
                res['time'] = np.concatenate((res['time'], time_array[mask]))
                res['Bmso'] = np.vstack((res['Bmso'], mat_data['Bmso'][np.squeeze(mask),:]))
                res['Pmso'] = np.vstack((res['Pmso'], mat_data['Pmso'][np.squeeze(mask),:]))
                
                # convert the data to TSeries
        B = ts_vec_xyz(res['time'], res['Bmso'], attrs={"name": "Magnetic field",
                                                                             "instrument": "MAG",
                                                                             "UNITS":"nT",
                                                                             "coordinates":"MSO",
                                                                             "resolution":"32Hz"})
        P = ts_vec_xyz(res['time'], res['Pmso'], attrs={"name": "Position",
                                                                             "UNITS":"km",
                                                                             "coordinates":"MSO",
                                                                             "resolution":"32Hz"})
        res = {'Bmso':B, 'Pmso':P}     
        return res

    #%% read the magnetic field (1Hz), MSO and MSE
    elif var == 'Bmse':
        res = {'time': np.array([], dtype='datetime64[ns]'),
               'Bmso': np.array([], dtype=float).reshape(0, 3),
               'Pmso': np.array([], dtype=float).reshape(0, 3),              
               'Bmse': np.array([], dtype=float).reshape(0, 3),
               'Pmse': np.array([], dtype=float).reshape(0, 3),
               'In_IMF':np.array([], dtype=float).reshape(0, 3),
               'Out_IMF':np.array([], dtype=float).reshape(0, 3),
               'In_Vsw':np.array([], dtype=float).reshape(0, 3),
               'Out_Vsw':np.array([], dtype=float).reshape(0, 3),
               'In_Nsw':np.array([], dtype=float).reshape(0, 1),
               'Out_Nsw':np.array([], dtype=float).reshape(0, 1),
               'In_Pdy':np.array([], dtype=float).reshape(0, 1),
               'Out_Pdy':np.array([], dtype=float).reshape(0, 1),
               'imf':np.array([], dtype=float).reshape(0, 3),
               'xmse':np.array([], dtype=float).reshape(0, 3),
               'ymse':np.array([], dtype=float).reshape(0, 3),
               'zmse':np.array([], dtype=float).reshape(0, 3),
               'len':np.array([], dtype=float).reshape(0, 1)}
        file_pattern = os.path.join(config['path'], config['filename_format'])
        filelist = glob.glob(file_pattern)
        for filename in filelist:
            file_start, file_end = read_time_from_file(filename)
            if (file_start <= np.datetime64(end_time)) and (file_end >= np.datetime64(start_time)):
                mat_data = loadmat(filename)
                time_array = irf_time(mat_data['time'],"datenum>datetime64")
                mask = (time_array >= start_time) & (time_array <= end_time)
                res['time'] = np.concatenate((res['time'], time_array[mask]))
                res['Bmso'] = np.vstack((res['Bmso'], mat_data['Bmso'][np.squeeze(mask),:]))
                res['Pmso'] = np.vstack((res['Pmso'], mat_data['Pmso'][np.squeeze(mask),:]))
                res['Bmse'] = np.vstack((res['Bmse'], mat_data['Bmse'][np.squeeze(mask),:]))
                res['Pmse'] = np.vstack((res['Pmse'], mat_data['Pmse'][np.squeeze(mask),:]))
                res['In_Vsw']=np.vstack((res['In_Vsw'],  mat_data['In_Vsw']))
                res['In_Nsw']=np.vstack((res['In_Nsw'],  mat_data['In_Nsw']))
                res['In_IMF']=np.vstack((res['In_IMF'],  mat_data['In_IMF']))
                res['In_Pdy']=np.vstack((res['In_Pdy'],  mat_data['In_Pdy']))
                res['Out_Vsw']=np.vstack((res['Out_Vsw'],mat_data['Out_Vsw']))
                res['Out_Nsw']=np.vstack((res['Out_Nsw'],mat_data['Out_Nsw']))
                res['Out_IMF']=np.vstack((res['Out_IMF'],mat_data['Out_IMF']))
                res['Out_Pdy']=np.vstack((res['Out_Pdy'],mat_data['Out_Pdy']))
                res['imf']=np.vstack((res['imf'],  mat_data['imf']))
                res['xmse']=np.vstack((res['xmse'],mat_data['xmse']))
                res['ymse']=np.vstack((res['ymse'],mat_data['ymse']))
                res['zmse']=np.vstack((res['zmse'],mat_data['zmse']))
                res['len']=np.vstack((res['len'],np.size(mat_data['time'][mask])))
        Bmso = ts_vec_xyz(res['time'], res['Bmso'], attrs={"name": "Magnetic field",
                                                                                     "instrument": "MAG",
                                                                                     "UNITS":"nT",
                                                                                     "coordinates":"MSO",
                                                                                     "resolution":"1Hz"})
        Pmso = ts_vec_xyz(res['time'], res['Pmso'], attrs={"name": "Position",
                                                                                     "UNITS":"km",
                                                                                     "coordinates":"MSO",
                                                                                     "resolution":"1Hz"})
        Bmse = ts_vec_xyz(res['time'], res['Bmse'], attrs={"name": "Magnetic field",
                                                                                     "instrument": "MAG",
                                                                                     "UNITS":"nT",
                                                                                     "coordinates":"MSE",
                                                                                     "resolution":"1Hz"})
        Pmse = ts_vec_xyz(res['time'], res['Pmse'], attrs={"name": "Position",
                                                                                     "UNITS":"km",
                                                                                     "coordinates":"MSE",
                                                                                     "resolution":"1Hz"})
        res = {'Bmso':Bmso, 'Pmso':Pmso, 'Bmse':Bmse, 'Pmse':Pmse,
               'In_Vsw':res['In_Vsw'], 'In_Nsw':res['In_Nsw'], 'In_IMF':res['In_IMF'], 'In_Pdy':res['In_Pdy'],
               'Out_Vsw':res['Out_Vsw'], 'Out_Nsw':res['Out_Nsw'], 'Out_IMF':res['Out_IMF'], 'Out_Pdy':res['Out_Pdy'],
               'imf':res['imf'], 'xmse':res['xmse'],'ymse':res['ymse'], 'zmse':res['zmse'],'len':res['len']}  
        return res

    #%% read the SWIA omni data and onboardmoment
    elif var == 'swia_omni':
        res={  'time': np.array([], dtype='datetime64[ns]'),
               'time_def': np.array([], dtype='datetime64[ns]'),
               'N': np.array([], dtype=float),
               'Vmso': np.array([], dtype=float).reshape(0, 3),
               'Vswia': np.array([], dtype=float).reshape(0, 3),
               'Temp': np.array([], dtype=float).reshape(0, 3), 
               'energy':np.array([], dtype=float),
               'DEF':np.array([], dtype=float).reshape(0, 48)}
        for date_file in dates_to_read:            
            year, month, day= year_month_day(date_file) 
            
            # the format of the name of Moment data CDF file
            file_mom_pattern = os.path.join(config['path'], year, month, 
                                        config['filename_mom_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
           
            # the format of the name of Omni DEF data CDF file
            file_def_pattern = os.path.join(config['path'], year, month, 
                                        config['filename_def_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
           
            # filename of the moment data and Omni DEF data
            filename_mom = glob.glob(file_mom_pattern)[0]
            filename_def = glob.glob(file_def_pattern)[0]
            
            # get the time, density, velocity, and temperature from the moment data
            val_names = ['epoch','density','velocity_mso', 'velocity','temperature_mso']
            is_time = [1] + [0] * (len(val_names) - 1)

            time_temp_mom, N_temp, \
                V_temp, V_temp_swia, Temp_temp = get_cdf_var(filename_mom, 
                                                             variable_name = val_names, 
                                                             istime = is_time)
            mask_mom = (time_temp_mom >= np.datetime64(start_time)) & (time_temp_mom <= np.datetime64(end_time))
            
            # get the time, density, velocity, and temperature from the moment data
            time_temp_def, energy_temp, DEF_temp = get_cdf_var(filename_def, 
                                        ['epoch','energy_spectra','spectra_diff_en_fluxes'],
                                        istime  = [1, 0, 0])
            mask_def = (time_temp_def >= np.datetime64(start_time)) & (time_temp_def <= np.datetime64(end_time))
           
            # construct the data
            res['time'] = np.concatenate((res['time'], time_temp_mom[mask_mom]))
            res['N']    = np.concatenate((res['N']   , N_temp[mask_mom]))
            res['Vmso'] = np.vstack((res['Vmso'], V_temp[np.squeeze(mask_mom),:]))
            res['Vswia'] = np.vstack((res['Vswia'], V_temp_swia[np.squeeze(mask_mom),:]))
            res['Temp'] = np.vstack((res['Temp'], Temp_temp[np.squeeze(mask_mom),:]))
            res['time_def'] = np.concatenate((res['time_def'], time_temp_def[mask_def]))
            res['energy'] = energy_temp
            res['DEF'] = np.vstack((res['DEF'], DEF_temp[np.squeeze(mask_def),:]))
            
       # convert the data to TSeries
        N = ts_scalar(res['time'], res['N'], attrs={"name":"SWIA_onboard_density",
                                                              "Instrument": "SWIA",
                                                                     "UNITS":"cm^-3"})
        Vmso = ts_vec_xyz(res['time'], res['Vmso'], attrs={"name":"SWIA_onboard_velocity",
                                                              "Instrument": "SWIA",
                                                                     "UNITS":"km/s",
                                                                     "coordinates":"MSO"})
        Vswia = ts_vec_xyz(res['time'], res['Vswia'], attrs={"name":"SWIA_onboard_velocity",
                                                              "Instrument": "SWIA",
                                                                     "UNITS":"km/s",
                                                                     "coordinates":"SWIA"})
        Temp = ts_vec_xyz(res['time'], res['Temp'], attrs={"name":"SWIA_onboard_temperature",
                                                              "Instrument": "SWIA",
                                                                     "UNITS":"K",
                                                                     "coordinates":"MSO"})

        omni_flux = ts_spectr(
            time=res['time_def'],  # 时间维度（1D array, datetime64）
            ener=np.flip(res['energy']),  # 能量维度（翻转顺序，从高能到低能）
            data=np.flip(res['DEF'], axis=1),  # 数据体（二维矩阵，需与时间和能量匹配）
            comp_name="energy",  # 能量轴的维度名
            attrs={
                "name": "SWIA_omni_eflux",  # 可选元信息
                "Instrument": "SWIA",
                "UNITS": "keV/(cm^2 s sr keV)",
            }
        )

        res = {'N':N,
               'Vmso':Vmso,
               'Vswia':Vswia,
               'Temp':Temp,
               'omni_flux':omni_flux}
        return res

    # %% read the Swia 3d data
    elif var == 'swia_3d':
        nenergy, nphi, ntheta = 48, 16, 4
        res = {
            'time': np.array([], dtype='datetime64[ns]'),
            'energy': np.empty((0, nenergy)),
            'DEF': np.empty((0, nenergy, nphi, ntheta)),
            'phi':np.array([], dtype=np.float64),
            'theta':np.array([], dtype=np.float64),
        }
        for date_file in dates_to_read:
            year, month, day = year_month_day(date_file)
            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month,
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
            filename = glob.glob(file_pattern)[0]
            data = swia.read_swia_3d(filename)
            res['time'] = np.concatenate((res['time'], data['time']))
            res['DEF'] = np.vstack((res['DEF'], data['DEF']))

        time, DEF = tint_data(res["time"],
                              np.datetime64(start_time),  np.datetime64(end_time),
                              res["DEF"])

        swia_vdf = create_pdist_skymap(time, data["energy"],
                                       DEF, data["phi"], data["theta"],
                                       Units="keV/(cm^2 s sr keV)",
                                       species="H+",
                                       direction_is_velocity=True, )
        return swia_vdf


    #%% read the SWEA omni data
    elif var == 'swea_omni':
        res={  'time': np.array([], dtype='datetime64[ns]'),
               'energy':np.array([], dtype=float),
               'DEF':np.array([], dtype=float).reshape(0, 64)}
        
        for date_file in dates_to_read:            
            year, month, day= year_month_day(date_file) 
           
            # the format of SWEA CDF file
            file_pattern = os.path.join(config['path'], year, month, 
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
          
            # filename of the SWEA DEF data
            filename = glob.glob(file_pattern)[0]
            time_temp, energy_temp, DEF_temp   = get_cdf_var(filename,
                                                             ['epoch', 'energy','diff_en_fluxes'],
                                                             istime= [1, 0, 0]) 
            mask = (time_temp >= np.datetime64(start_time)) & (time_temp <= np.datetime64(end_time))
           
            # construct the data
            res['time'] = np.concatenate((res['time'], time_temp[mask]))
            res['energy'] = energy_temp
            res['DEF'] = np.vstack((res['DEF'], DEF_temp[np.squeeze(mask),:]))
            
       # convert the data to TSeries
        swea = ts_spectr(
            time=res['time'],
            ener = np.flip(res['energy']),
            data=np.flip(res['DEF'], axis=1),
            comp_name="energy",
            attrs={
                "name": "SWEA_omni_eflux",
                "Instrument": "SWEA",
                "UNITS": "keV/(cm^2 s sr keV)",
            }
        )
        return swea

    # %% read the SWEA omni data
    elif var == 'swea_scpot':
        res = {'time': np.array([], dtype='datetime64[ns]'),
               'scpot': np.array([], dtype=float),}

        for date_file in dates_to_read:
            filename = os.path.join(config['path'], config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
            if os.path.exists(filename):
                mat_data = loadmat(filename)
                time_array = irf_time(mat_data['time'],"datenum>datetime64")

                # select the data within the interval
                mask = (time_array >= np.datetime64(start_time)) & (time_array <= np.datetime64(end_time))
                res['time'] = np.concatenate((res['time'], time_array[mask]))
                res['scpot'] = np.concatenate((res['scpot'], mat_data['scpot'][mask]))

        # convert the data to TSeries
        res = ts_scalar(time=res['time'], data=res['scpot'])
        return res

    #%% read the SWEA pad data
    elif var == 'swea_pad':
        res={  'time': np.array([], dtype='datetime64[ns]'),
               'energy':np.array([], dtype=float),
               'DEF':np.array([], dtype=float).reshape(0, 16, 64),
               'PA':np.array([], dtype=float).reshape(0, 16, 64),
               'bazim':np.array([], dtype=float),
               'belev':np.array([], dtype=float),}
        for date_file in dates_to_read:            
            year, month, day= year_month_day(date_file) 
           
            # the format of SWEA CDF file
            file_pattern = os.path.join(config['path'], year, month, 
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
          
            # filename of the SWEA PAD data
            filename = glob.glob(file_pattern)[0]
            val_names = ['epoch', 'energy', 'diff_en_fluxes', 'pa', 'b_azim', 'b_elev']
            is_time = [1] + [0] * (len(val_names) - 1)
            
            time_temp, energy_temp, DEF_temp, \
                PAD_temp, bazim_temp, belev_temp = get_cdf_var(cdf_filename = filename,
                                                               variable_name = val_names,
                                                               istime = is_time) 

            mask = (time_temp >= np.datetime64(start_time)) & (time_temp <= np.datetime64(end_time))
            
            # construct the data
            res['time'] = np.concatenate((res['time'], time_temp[mask]))
            res['energy'] = energy_temp
            res['DEF'] = np.vstack((res['DEF'], DEF_temp[np.squeeze(mask),:,:]))
            res['PA'] = np.vstack((res['PA'], PAD_temp[np.squeeze(mask),:,:]))
            res['bazim'] = np.concatenate((res['bazim'], bazim_temp[mask]))
            res['belev'] = np.concatenate((res['belev'], belev_temp[mask]))
        res['DEF'] = np.transpose(res['DEF'], (0, 2, 1))
        res['PA'] = np.transpose(res['PA'], (0, 2, 1))

        return res
    
    #%% read the Static c0 data
    elif var == 'static_c0':
        nenergy, nmass = 64, 2
        res = {
            'time': np.array([], dtype='datetime64[ns]'),
            'energy': np.empty((0, nenergy)),
            'DEF': np.empty((0, nenergy, nmass)),
            # 'count': np.empty((0, nenergy, nmass)),
            # 'Bdmpa': np.empty((0, 3)),
            'scpot': np.array([], dtype=float),
            }
        
        for date_file in dates_to_read:            
            year, month, day= year_month_day(date_file) 
           
            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month, 
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
          
            filename = glob.glob(file_pattern)[0]
            data = static.read_c0(filename)
            mask = (data['time'] >= np.datetime64(start_time)) & (data['time'] <= np.datetime64(end_time))
            res['time']   = np.concatenate((res['time'], data['time'][mask]))
            res['energy'] = np.vstack((res['energy'], data['energy'][mask, :]))
            res['DEF']    = np.vstack((res['DEF'], data['DEF'][mask, :, :]))
            # res['count']  = np.vstack((res['count'], data['count'][mask, :, :]))
            res['scpot']  = np.concatenate((res['scpot'], data['scpot'][mask]))
            scpot = ts_scalar(res["time"], res['scpot'], attrs={"name":"Spacecraft potential",
                                                                      "UNITS":"eV"} )
            res['scpot'] = scpot
            res['mass']   = data['mass']
        return res
        
    #%% read the Static c6 data
    elif var == 'static_c6':
        nenergy, nmass = 32, 64
        res = {
            'time': np.array([], dtype='datetime64[ns]'),
            'energy': np.empty((0, nenergy)),
            'DEF': np.empty((0, nenergy, nmass)),
            'dead': np.empty((0, nenergy, nmass)),
            'count': np.empty((0, nenergy, nmass)),
            'scpot': np.array([], dtype=float),
            'denergy': np.empty((0, nenergy)),
            'dtheta': np.empty((0, nenergy)),
            }
        
        for date_file in dates_to_read:            
            year, month, day= year_month_day(date_file) 
           
            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month, 
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
          
            filename = glob.glob(file_pattern)[0]
            data = static.read_c6(filename)
            mask = (data['time'] >= np.datetime64(start_time)) & (data['time'] <= np.datetime64(end_time))
            res['time']   = np.concatenate((res['time'], data['time'][mask]))
            res['energy'] = np.vstack((res['energy'], data['energy'][mask, :]))
            res['denergy'] = np.vstack((res['denergy'], data['denergy'][mask, :]))
            res['dtheta'] = np.vstack((res['dtheta'], data['dtheta'][mask, :]))
            res['DEF']    = np.vstack((res['DEF'], data['DEF'][mask, :, :]))
            res['dead']    = np.vstack((res['dead'], data['dead'][mask, :, :]))
            res['count']    = np.vstack((res['count'], data['count'][mask, :, :]))
            res['scpot']  = np.concatenate((res['scpot'], data['scpot'][mask]))
            res['mass']   = data['mass']
            scpot = ts_scalar(res["time"], res['scpot'], attrs={"name":"Spacecraft potential",
                                                                      "UNITS":"eV"} )
            res['scpot'] = scpot
        return res

    # %% read the Static c8 data
    elif var == 'static_c8':
        nenergy, ndirection = 32, 16
        res = {
            'time': np.array([], dtype='datetime64[ns]'),
            'energy': np.empty((0, nenergy)),
            'DEF': np.empty((0, nenergy, ndirection)),
            'count': np.empty((0, nenergy, ndirection)),
            'scpot': np.array([], dtype=float),
        }

        for date_file in dates_to_read:
            year, month, day = year_month_day(date_file)

            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month,
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))

            filename = glob.glob(file_pattern)[0]
            data = static.read_c8(filename)
            mask = (data['time'] >= np.datetime64(start_time)) & (data['time'] <= np.datetime64(end_time))
            res['time'] = np.concatenate((res['time'], data['time'][mask]))
            res['energy'] = np.vstack((res['energy'], data['energy'][mask, :]))
            res['DEF'] = np.vstack((res['DEF'], data['DEF'][mask, :, :]))
            res['count']  = np.vstack((res['count'], data['count'][mask, :, :]))
            res['scpot'] = np.concatenate((res['scpot'], data['scpot'][mask]))
            scpot = ts_scalar(res["time"], res['scpot'], attrs={"name": "Spacecraft potential",
                                                                "UNITS": "eV"})
            res['scpot'] = scpot
        return res

    # %% read the Static c6 iv4 data
    elif var == 'static_c6_iv4':
        nenergy, nmass = 32, 64
        res = {
            'time': np.array([], dtype='datetime64[ns]'),
            'time_integ': np.array([], dtype=float),
            'bkg': np.empty((0, nenergy, nmass)),
            'gf': np.empty((0, nenergy, nmass)),
        }
        for date_file in dates_to_read:
            year, month, day = year_month_day(date_file)

            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month,
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))

            filename = glob.glob(file_pattern)[0]
            data = static.read_c6_iv4(filename)
            mask = (data['time'] >= np.datetime64(start_time)) & (data['time'] <= np.datetime64(end_time))
            res['time'] = np.concatenate((res['time'], data['time'][mask]))
            res['time_integ'] = np.concatenate((res['time_integ'], data['time_integ'][mask]))
            res['bkg'] = np.vstack((res['bkg'], data['bkg'][mask, :, :]))
            res['gf'] = np.vstack((res['gf'], data['gf'][mask, :, :]))
        return res

    #%% read the Static d1 data
    elif var == 'static_d1':
        nenergy, nphi, ntheta, nmass = 32, 16, 4, 8 
        res = {
            'time': np.array([], dtype='datetime64[ns]'),
            'energy': np.empty((0, nenergy)),
            'count': np.empty((0, nenergy, nphi, ntheta, nmass)),
            'dead': np.empty((0, nenergy, nphi, ntheta, nmass)),
            'H_DEF': np.empty((0, nenergy, nphi, ntheta)),
            'O_DEF': np.empty((0, nenergy, nphi, ntheta)),
            'O2_DEF': np.empty((0, nenergy, nphi, ntheta)),
            'CO2_DEF': np.empty((0, nenergy, nphi, ntheta)),
            'Bdmpa': np.empty((0, 3)),
            'scpot': np.array([], dtype=float),
            'theta': np.empty((0, nenergy, ntheta)),
            'dtheta': np.empty((0, nenergy, ntheta)),
            'sta2mso': np.empty((0, 3, 3)),
            }
        
        for date_file in dates_to_read:            
            year, month, day= year_month_day(date_file) 
           
            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month, 
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
          
            filename = glob.glob(file_pattern)[0]
            data = static.read_d1(filename)
            
            mask = (data['time'] >= np.datetime64(start_time)) & (data['time'] <= np.datetime64(end_time))
            res['time']   = np.concatenate((res['time'], data['time'][mask]))
            res['dtheta'] = np.vstack((res['dtheta'], data['dtheta'][mask,:,:]))
            res['theta'] = np.vstack((res['theta'], data['theta'][mask,:,:]))
            res['H_DEF']    = np.vstack((res['H_DEF'], data['H_DEF'][mask]))
            res['O_DEF']    = np.vstack((res['O_DEF'], data['O_DEF'][mask]))
            res['O2_DEF']    = np.vstack((res['O2_DEF'], data['O2_DEF'][mask]))
            res['CO2_DEF']    = np.vstack((res['CO2_DEF'], data['CO2_DEF'][mask]))
            res['count']  = np.vstack((res['count'], data['count'][mask]))
            res['dead']  = np.vstack((res['dead'], data['dead'][mask]))
            res['Bdmpa']  = np.vstack((res['Bdmpa'], data['Bdmpa'][mask,:]))
            res['scpot']  = np.concatenate((res['scpot'], data['scpot'][mask]))
            res['energy'] = np.vstack((res['energy'], data['energy'][mask]))
            res['sta2mso'] = np.vstack((res['sta2mso'], data['sta2mso'][mask]))
            res["phi"] = data["phi"]
            H_DEF = create_pdist_skymap(res["time"], res["energy"],
                                                           res["H_DEF"], res["phi"], res["theta"],
                                                           Units="keV/(cm^2 s sr keV)",
                                                           species="H+",
                                                           direction_is_velocity=True,
                                                           deltatheta = res['dtheta'],)
            O_DEF = create_pdist_skymap(res["time"], res["energy"],
                                                           res["O_DEF"], res["phi"], res["theta"],
                                                           Units="keV/(cm^2 s sr keV)",
                                                           species="O+",
                                                           direction_is_velocity=True,
                                                           deltatheta = res['dtheta'],)
            O2_DEF = create_pdist_skymap(res["time"], res["energy"],
                                                           res["O2_DEF"], res["phi"], res["theta"],
                                                           Units="keV/(cm^2 s sr keV)",
                                                           species="O2+",
                                                           direction_is_velocity=True,
                                                           deltatheta = res['dtheta'],)
            CO2_DEF = create_pdist_skymap(res["time"], res["energy"],
                                                           res["CO2_DEF"], res["phi"], res["theta"],
                                                           Units="keV/(cm^2 s sr keV)",
                                                           species="O2+",
                                                           direction_is_velocity=True,
                                                           deltatheta = res['dtheta'],)
            Bdmpa = ts_vec_xyz(res["time"], res["Bdmpa"], attrs={"name":"B in STATIC Frame",
                                                                      "UNITS":"nT",
                                                                      "coordinates":"STATIC"} )
            scpot = ts_scalar(res["time"], res['scpot'], attrs={"name":"Spacecraft potential",
                                                                      "UNITS":"eV"} )
            res["H_DEF"] = H_DEF
            res["O_DEF"] = O_DEF
            res["O2_DEF"] = O2_DEF
            res["CO2_DEF"] = CO2_DEF
            res["Bdmpa"] = Bdmpa
            res["scpot"] = scpot
            for key in ["energy", "theta", "dtheta", "phi"]:
                del res[key]
        return res


    # %% read the Static d1 iv4 data
    elif var == 'static_d1_iv4':
        nenergy, nphi, ntheta, nmass = 32, 16, 4, 8
        res = {
            'time': np.array([], dtype='datetime64[ns]'),
            'time_integ': np.array([], dtype=float),
            'bkg': np.empty((0, nenergy, nphi, ntheta, nmass)),
            'gf': np.empty((0, nenergy, nphi, ntheta, nmass)),
        }
        for date_file in dates_to_read:
            year, month, day = year_month_day(date_file)

            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month,
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))

            filename = glob.glob(file_pattern)[0]
            data = static.read_d1_iv4(filename)
            mask = (data['time'] >= np.datetime64(start_time)) & (data['time'] <= np.datetime64(end_time))
            res['time'] = np.concatenate((res['time'], data['time'][mask]))
            res['time_integ'] = np.concatenate((res['time_integ'], data['time_integ'][mask]))
            res['bkg'] = np.vstack((res['bkg'], data['bkg'][mask, :, :, :, :]))
            res['gf'] = np.vstack((res['gf'], data['gf'][mask, :, :, :, :]))
        return res

    # %% read the static density data, Gwen
    elif var == 'static_density':
        res = {'time': np.array([], dtype='datetime64[ns]'),
               'nH': np.array([], dtype=float),
               'nHe': np.array([], dtype=float),
               'nO': np.array([], dtype=float),
               'nO2': np.array([], dtype=float),}
        for date_file in dates_to_read:
            filename = os.path.join(config['path'],
                                    config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
            if os.path.exists(filename):
                mat_data = static.read_gwen_density_txt(filename)
                time_array = mat_data['time']
                # select the data within the interval
                mask = (time_array >= np.datetime64(start_time)) & (time_array <= np.datetime64(end_time))
                res['time'] = np.concatenate((res['time'], time_array[mask]))
                res['nH'] = np.concatenate((res['nH'], mat_data['nH'][mask]))
                res['nHe'] = np.concatenate((res['nHe'], mat_data['nHe'][mask]))
                res['nO'] = np.concatenate((res['nO'], mat_data['nO'][mask]))
                res['nO2'] = np.concatenate((res['nO2'], mat_data['nO2'][mask]))
                
        res['nH'] = ts_scalar(res['time'], res['nH'], attrs={"name":"Gwen_density",
                                                              "Instrument": "STATIC",
                                                                     "UNITS":"cm^-3",
                                                                     "species":"H+"})
        res['nHe'] = ts_scalar(res['time'], res['nHe'], attrs={"name":"Gwen_density",
                                                              "Instrument": "STATIC",
                                                                     "UNITS":"cm^-3",
                                                                     "species":"He++"})
        res['nO'] = ts_scalar(res['time'], res['nO'], attrs={"name":"Gwen_density",
                                                              "Instrument": "STATIC",
                                                                     "UNITS":"cm^-3",
                                                                     "species":"O+"})
        res['nO2'] = ts_scalar(res['time'], res['nO2'], attrs={"name":"Gwen_density",
                                                              "Instrument": "STATIC",
                                                                     "UNITS":"cm^-3",
                                                                     "species":"O2+"})

        return res


    # %% read the static moment data made by Chi Zhang
    elif var == "static_moment":
        res = {'time': np.array([], dtype='datetime64[ns]'),
               'nH': np.array([], dtype=float),
               'vH': np.empty((0, 3), dtype=float),
               'nO': np.array([], dtype=float),
               'vO': np.empty((0, 3), dtype=float),
               'nO2': np.array([], dtype=float),
               'vO2': np.empty((0, 3), dtype=float),
               'sun_el': np.array([], dtype=float),}

        for date_file in dates_to_read:
            year, month, day = year_month_day(date_file)
            # the format of CDF file
            file_pattern = os.path.join(config['path'], year, month,
                                        config['filename_format'].format(date=date_file.astype('O').strftime('%Y%m%d')))
            filename = glob.glob(file_pattern)[0]
            if os.path.exists(filename):
                mat_data = loadmat(filename)
                time_array = irf_time(mat_data['time'],"datenum>datetime64")
                # select the data within the interval
                mask = (time_array >= np.datetime64(start_time)) & (time_array <= np.datetime64(end_time))
                res['time'] = np.concatenate((res['time'], time_array[mask]))
                res['vH'] = np.vstack((res['vH'], mat_data['vH'][np.squeeze(mask),:]))
                res['vO'] = np.vstack((res['vO'], mat_data['vO'][np.squeeze(mask),:]))
                res['vO2'] = np.vstack((res['vO2'], mat_data['vO2'][np.squeeze(mask), :]))
                res['nH'] = np.concatenate((res['nH'], mat_data['nH'][mask]))
                res['nO'] = np.concatenate((res['nO'], mat_data['nO'][mask]))
                res['nO2'] = np.concatenate((res['nO2'], mat_data['nO2'][mask]))
                res['sun_el'] = np.concatenate((res['sun_el'], mat_data['sun_el'][mask]))
        
        res['nH'] = ts_scalar(res['time'], res['nH'], attrs={"name":"Gwen_density",
                                                              "Instrument": "STATIC",
                                                                     "UNITS":"cm^-3",
                                                                     "species":"H+"})
        res['nO'] = ts_scalar(res['time'], res['nO'], attrs={"name":"Gwen_density",
                                                              "Instrument": "STATIC",
                                                                     "UNITS":"cm^-3",
                                                                     "species":"O+"})
        res['nO2'] = ts_scalar(res['time'], res['nO2'], attrs={"name":"Gwen_density",
                                                              "Instrument": "STATIC",
                                                                     "UNITS":"cm^-3",
                                                                     "species":"O2+"})
        res['vO'] = ts_vec_xyz(res['time'], res['vO'], attrs={"name":"STATIC_velocity",
                                                              "Instrument": "STATIC",
                                                              "UNITS":"km/s",
                                                              "coordinates":"MSO",
                                                              "species":"O+"})
        res['vH'] = ts_vec_xyz(res['time'], res['vH'], attrs={"name":"STATIC_velocity",
                                                              "Instrument": "STATIC",
                                                              "UNITS":"km/s",
                                                              "coordinates":"MSO",
                                                              "species":"H+"})
        res['vO2'] = ts_vec_xyz(res['time'], res['vO2'], attrs={"name":"STATIC_velocity",
                                                              "Instrument": "STATIC",
                                                              "UNITS":"km/s",
                                                              "coordinates":"MSO",
                                                              "species":"O2+"})
        
        return res




def load_data(tint, var_list):
    """
    Load one or more MAVEN data variables for the given time interval.

    Parameters
    ----------
    tint : list of str
        Time interval in the format ['YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DDTHH:MM:SS'].
    var_list : str or list of str
        Variable name(s) to load. Can be a single string or a list of strings.

    Returns
    -------
    object or tuple of objects
        - If a single variable name is provided: returns the corresponding data object.
        - If a list of variable names is provided: returns a tuple of data objects in the same order.

    Examples
    --------
    >>> swea = load_data(tint, 'swea_omni')
    >>> b, swia = load_data(tint, ['B', 'swia_omni'])

    Notes
    -----
    Internally calls `get_data(tint, var)` for each variable name in var_list.
    """
    if isinstance(var_list, str):
        return get_data(tint, var_list)
    elif isinstance(var_list, (list, tuple)):
        return tuple(get_data(tint, var) for var in var_list)
    else:
        raise TypeError("var_list must be a string, list, or tuple.")




if __name__ == '__main__':
    tint = ["2023-07-31T04:38:10", "2023-07-31T04:40:05"]
    scpot = get_data(tint, 'swea_scpot')
    # B, swia_omni, swia_3d = load_data(tint, ['B', 'swia_omni', 'swia_3d'])
    # swea_omni = load_data(tint, ['swea_omni'])







    



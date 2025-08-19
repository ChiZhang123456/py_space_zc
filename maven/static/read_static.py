"""
Created on Thu Jul 17 10:02:25 2025
Author: Chi Zhang

This script provides functions to read MAVEN STATIC plasma data products
from CDF files using the py_space_zc interface (which wraps `get_cdf_var`).
It supports three STATIC modes:

    - read_c0: For C0 mode (64 energy bins × 2 mass bins)
    - read_c6: For C6 mode (32 energy bins × 64 mass bins)
    - read_d1: For D1 mode (32 energy bins × 16 phi x 4 theta × 8 masses)
"""

import os
import numpy as np
from py_space_zc import get_cdf_var

#%% STATIC C0 mode reader
def read_c0(filename_c0):
    """
    Read and process MAVEN STATIC C0 data from a CDF file.

    Parameters
    ----------
    filename : str
        Full path to the CDF file.

    Returns
    -------
    res : dict
        Dictionary with the following fields:
        - time      [num_time]                            : Time stamps in datetime64
        - energy    [num_time, num_energy]                : Energy table 
        - DEF       [num_time, num_energy, num_mass]      : Differential energy flux
        - Bdmpa     [num_time, 3]                         : Magnetic field in STATIC DMPA frame
        - scpot     [num_time]                            : Spacecraft potential
        - mass      [num_mass]      : Mass table
    """
    nenergy = 64
    nmass = 2 

    # Initialize empty result dictionary
    res = {
        'time': np.array([], dtype='datetime64[ns]'),
        'energy': np.array([], dtype=float),
        'DEF': np.array([], dtype=float).reshape(0, nenergy, nmass),
        'Bdmpa': np.array([], dtype=float),
        'scpot': np.array([], dtype=float),
        'mass': np.array([], dtype=float),
    }

    # Return empty result if file is missing
    if not os.path.isfile(filename_c0):
        print(f"File not found: {filename_c0}")
        return res

    try:
         # Read CDF variables
        val_names = ['epoch','energy','sc_pot','eflux','magf','mass_arr','swp_ind']
        is_time = [1] + [0] * (len(val_names) - 1)
        
        time, energy, scpot, DEF, Bdmpa,\
            mass_arr, swp_ind = get_cdf_var(cdf_filename = filename_c0, 
                                            variable_name = val_names,
                                            istime = is_time)
       # energy : [nmass, nenergy, nswp]
       # DEF:     [ntime, nmass, nenergy]
       # mass:    [nmass, nenergy, nswp]
        swp_ind = swp_ind.astype(int)

        # Format and transpose
        DEF = np.transpose(DEF, [0, 2, 1])                        # [ntime, nenergy, nmass]
        mass_arr = np.transpose(mass_arr, [1, 0, 2])              # [nenergy, nmass, nswp]
        energy = np.squeeze(energy[0, :, :])                      # [nenergy, nswp]
        
        # Initialize per-timestep arrays
        temp_energy = np.zeros((len(time), nenergy))            # [ntime, nenergy]
        temp_mass   = np.zeros((len(time), nenergy, nmass))         # [ntime, nenergy, nmass]

        # Populate dynamic values based on sweep index
        for i in range(len(time)):
            swp_idx = swp_ind[i]
            temp_energy[i, :] = energy[:, swp_idx]           #  energy table
            temp_mass[i, :, :] = np.squeeze(mass_arr[:, :, swp_idx]) # Mass array 
        
        temp_mass = np.mean(temp_mass, (0, 1))
        # Store results (flipped energy axis: high to low)
        res['time'] = time
        res['energy'] = np.flip(temp_energy, 1)
        res['DEF'] = np.flip(DEF, 1)
        res['mass'] = temp_mass
        res['Bdmpa'] = Bdmpa
        res['scpot'] = scpot


    except Exception as e:
        # Optional: print a warning and return empty structure
        print(f"Warning: Failed to read {filename_c0}. Reason: {e}")

    return res

#%% STATIC C6 mode reader
def read_c6(filename_c6):
    """
    Read and process MAVEN STATIC C6 data from a CDF file.

    Parameters
    ----------
    filename : str
        Full path to the CDF file.

    Returns
    -------
    res : dict
        Dictionary with the following fields:
        - time       [num_time]                           : Time array
        - energy     [num_time, num_energy]               : Energy table
        - DEF        [num_time, num_energy, num_mass]     : Differential energy flux
        - count      [num_time, num_energy, num_mass]     : Raw counts
        - Bdmpa      [num_time, 3]                        : Magnetic field in STATIC DMPA frame
        - scpot      [num_time]                           : Spacecraft potential
        - mass       [num_time, num_mass]                 : Mass table
        - dtheta     [num_time, num_energy]               : Delta elevation angle
        - denergy    [num_time, num_energy]               : Energy width
        - dead       [num_time, num_energy, num_mass]     : Dead time correction
    """
    nenergy = 32
    nmass = 64

    res = {
        'time': np.array([], dtype='datetime64[ns]'),
        'energy': np.array([], dtype=float),
        'DEF': np.array([], dtype=float).reshape(0, nenergy, nmass),
        'count': np.array([], dtype=float).reshape(0, nenergy, nmass),
        'Bdmpa': np.array([], dtype=float),
        'scpot': np.array([], dtype=float),
        'mass': np.array([], dtype=float),
        'denergy': np.array([], dtype=float),
        'dtheta': np.array([], dtype=float),
        'dead': np.array([], dtype=float),
    }

    if not os.path.isfile(filename_c6):
        print(f"File not found: {filename_c6}")
        return res

    try:
        # Read CDF variables
        val_names = ['epoch','energy','sc_pot','eflux','data','magf','mass_arr','swp_ind',
                     'dtheta','denergy','dead']
        is_time = [1] + [0] * (len(val_names) - 1)
        
        time, energy, scpot, DEF, count, Bdmpa, mass_arr,\
            swp_ind, dtheta, denergy, dead = get_cdf_var(cdf_filename=filename_c6,
                                                         variable_name=val_names,
                                                         istime=is_time)
        swp_ind = swp_ind.astype(int)

        # Energy:     [nmass, nenergy, nswp]
        # DEF:        [ntime, nmass, nenergy]
        # count:      [ntime, nmass, nenergy]
        # mass_arr:   [nmass, nenergy, nswp]
        # dtheta:     [1, nenergy, nswp]
        # denergy:    [1, nenergy, nswp]
        # dead:       [ntime, nenergy, nmass]

        # Reshape
        DEF = np.transpose(DEF, [0, 2, 1])                        # [ntime, 32, 64]
        count = np.transpose(count, [0, 2, 1])
        dead = np.transpose(dead, [0, 2, 1])
        energy = np.squeeze(energy[0, :, :])                     # [32, nswp]
        denergy = np.squeeze(denergy[0, :, :])                             # [32, nswp]
        dtheta = np.squeeze(dtheta[0, :, :])                               # [32, nswp]
        mass_arr = np.nanmean(mass_arr, axis=1)                            # [64, nswp]

        # Allocate output arrays
        temp_energy = np.zeros((len(time), nenergy))
        temp_mass = np.zeros((len(time), nmass))
        temp_dtheta = np.zeros((len(time), nenergy))
        temp_denergy = np.zeros((len(time), nenergy))

        for i in range(len(time)):
            swp_idx = swp_ind[i]
            temp_energy[i, :] = energy[:, swp_idx]
            temp_dtheta[i, :] = dtheta[:, swp_idx]
            temp_denergy[i, :] = denergy[:, swp_idx]
            temp_mass[i, :] = np.squeeze(mass_arr[:, swp_idx])

        # Store results (flipped energy axis)
        res['time'] = time
        res['energy'] = np.flip(temp_energy, 1)
        res['DEF'] = np.flip(DEF, 1)
        res['count'] = np.flip(count, 1)
        res['dead'] = np.flip(dead, 1)
        res['dtheta'] = np.flip(temp_dtheta, 1)
        res['denergy'] = np.flip(temp_denergy, 1)
        res['Bdmpa'] = Bdmpa
        res['scpot'] = scpot
        res['mass'] = np.mean(temp_mass, 0)


    except Exception as e:
        print(f"Warning: Failed to read {filename_c6}. Reason: {e}")

    return res


#%% STATIC C6 iv4 mode reader
def read_c6_iv4(filename_c6_iv4: str) -> dict:
    """
    Read and process MAVEN STATIC C6 iv4 data from a CDF file (fixed shapes).
    nenergy = 32, nmass = 64
    bkg      : (ntime, nmass, nenergy)  -> will be transposed to (ntime, nenergy, nmass) and energy-flipped
    eff      : (nenergy, neff, nmass)
    gf_table : (nenergy, nswp, natt)
    eff_ind, att_ind, swp_ind : (ntime,)
    geom_factor : scalar (fixed across time)
    time_integ  : (ntime,)

    Returns
    -------
    res : dict with keys
        - time       : (ntime,) np.datetime64[ns]
        - bkg        : (ntime, nenergy, nmass)  # energy ascending
        - time_integ : (ntime,)
        - gf         : (ntime, nenergy, nmass)  # effective geometric factor, energy ascending
    """
    nenergy, nmass = 32, 64
    res = {
        "time": np.array([], dtype="datetime64[ns]"),
        "bkg": np.empty((0, nenergy, nmass), dtype=float),
        "time_integ": np.array([], dtype=float),
        "gf": np.empty((0, nenergy, nmass), dtype=float),
    }

    if not os.path.isfile(filename_c6_iv4):
        print(f"[read_c6_iv4] File not found: {filename_c6_iv4}")
        return res

    try:
        var_names = [
            "epoch", "bkg", "eff", "gf",
            "eff_ind", "att_ind", "swp_ind",
            "geom_factor", "time_integ",
        ]
        is_time = [1] + [0] * (len(var_names) - 1)

        (time, bkg, eff, gf_table,
         eff_ind, att_ind, swp_ind,
         geom_factor, time_integ) = get_cdf_var(
            cdf_filename=filename_c6_iv4,
            variable_name=var_names,
            istime=is_time,
        )

        ntime = len(time)
        eff_ind = eff_ind.astype(int)
        att_ind = att_ind.astype(int)
        swp_ind = swp_ind.astype(int)

        # bkg: (ntime, nmass, nenergy) -> (ntime, nenergy, nmass), then flip energy to ascending
        bkg = np.transpose(bkg, (0, 2, 1))

        # Select per-record gf and eff
        temp_gf  = np.empty((ntime, nenergy), dtype=float)         # (ntime, nenergy)
        temp_eff = np.empty((ntime, nenergy, nmass), dtype=float)  # (ntime, nenergy, nmass)
        for i in range(ntime):
            temp_gf[i, :]     = gf_table[att_ind[i], :, swp_ind[i]]  # (ntime, nenergy)
            temp_eff[i, :, :] = np.transpose(np.squeeze(eff[:, :, eff_ind[i]]))              # (ntime, nenergy, nmass)
        # Broadcast gf along mass, then multiply by scalar geom_factor
        gf_3d  = temp_gf[:, :, None]              # (ntime, nenergy, 1)
        gf_eff = geom_factor * gf_3d * temp_eff   # scalar * (ntime, nenergy, nmass)

        # Keep energy axis consistent with bkg (ascending)
        gf_eff = np.flip(gf_eff, axis=1)
        bkg = np.flip(bkg, axis=1)

        res["time"] = time
        res["bkg"] = bkg
        res["time_integ"] = time_integ
        res["gf"] = gf_eff

    except Exception as e:
        print(f"[read_c6_iv4] Warning: failed to read '{filename_c6_iv4}': {e}")

    return res


#%% STATIC d1 mode reader
def read_d1(filename_d1):
    """
    Read and process MAVEN STATIC D1 data from a CDF file.

    Parameters
    ----------
    filename_d1 : str
        Full path to the STATIC D1 CDF file.

    Returns
    -------
    res : dict
        Dictionary containing processed STATIC D1 data with the following fields:
            - time       [num_time]                            : Time array in datetime64[ns]
            - energy     [num_time, num_energy]                : Energy table (sorted high -> low)
            - denergy    [num_time, num_energy]                : Energy bin width
            - H_DEF      [num_time, num_energy, num_phi, num_theta] : H+ differential energy flux
            - O_DEF      [num_time, num_energy, num_phi, num_theta] : O+ differential energy flux
            - O2_DEF     [num_time, num_energy, num_phi, num_theta] : O2+ differential energy flux
            - CO2_DEF    [num_time, num_energy, num_phi, num_theta] : CO2+ differential energy flux
            - theta      [num_time, num_energy, num_theta]     : Elevation angles
            - dtheta     [num_time, num_energy, num_theta]     : Elevation angle bin width
            - phi        [num_phi]                             : Azimuthal angles [0, 360)
            - dphi       [scalar]                              : Azimuthal bin width
            - Bdmpa      [num_time, 3]                         : Magnetic field in STATIC DMPA frame
            - scpot      [num_time]                            : Spacecraft potential
            - mass       [num_mass]                            : Mass table
    """

    # Define expected dimensions based on STATIC D1 configuration
    nenergy, nmass, nphi, ntheta = 32, 8, 16, 4

    # Initialize result dictionary with empty arrays and proper shapes
    res = {
        'time': np.array([], dtype='datetime64[ns]'),
        'energy': np.empty((0, nenergy)),
        'H_DEF': np.empty((0, nenergy, nphi, ntheta)),
        'O_DEF': np.empty((0, nenergy, nphi, ntheta)),
        'O2_DEF': np.empty((0, nenergy, nphi, ntheta)),
        'CO2_DEF': np.empty((0, nenergy, nphi, ntheta)),
        'count': np.empty((0, nenergy, nphi, ntheta, nmass)),
        'dead': np.empty((0, nenergy, nphi, ntheta, nmass)),
        'theta': np.empty((0, nenergy, ntheta)),
        'dtheta': np.empty((0, nenergy, ntheta)),
        'phi': np.array([], dtype=float),
        'Bdmpa': np.array([], dtype=float),
        'scpot': np.array([], dtype=float),
    }

    if not os.path.isfile(filename_d1):
        print(f"File not found: {filename_d1}")
        return res

    try:
        # === Step 1: Read all variables in one call ===
        val_names = [
            'epoch', 'energy', 'sc_pot', 'eflux', 'data', 'dead', 'magf',
            'mass_arr', 'swp_ind', 'dtheta',
            'phi',  'theta', 'quat_mso'
        ]
        is_time = [1] + [0] * (len(val_names) - 1)

        (
            time, energy, scpot, DEF, count, dead, Bdmpa,
            mass_arr, swp_ind, dtheta,
            phi, theta, quat_mso
        ) = get_cdf_var(
            cdf_filename = filename_d1,
            variable_name = val_names,
            istime = is_time
        )
            
            #Energy:   [n_mass, 64, n_energy, swp_ind]
            #denergy:  [n_mass, 64, n_energy, swp_ind]
            #count:    [n_time, n_mass, 64, n_energy]
            #dead:     [n_time, n_mass, 64, n_energy]
            #DEF:      [n_time, n_mass, 64, n_energy]
            #theta:    [n_mass, 64, n_energy, swp_ind]
            #dtheta:   [n_mass, 64, n_energy, swp_ind]
            #phi:      [n_mass, 64, n_energy, swp_ind]
            #mass_arr: [n_mass, 64, n_energy, swp_ind]
            #quat_mso: [n_time, 4]
            

        swp_ind = swp_ind.astype(int)
        
        # phi
        phi = np.squeeze(phi[0,:,0,0]).reshape(16, 4)[:,0]
        phi[phi <= 0] += 360                           # Convert to 0-360 range

        # mass
        mass = np.squeeze(np.nanmean(np.squeeze(mass_arr[:,1,:,:]), 2))[:,0]

        # theta
        nswp = theta.shape[-1]
        theta  = np.reshape(theta,  [nmass, nphi, ntheta, nenergy, nswp])[0,0,:,:,:].squeeze() # [ntheta, nenergy, nswp] 
        dtheta = np.reshape(dtheta, [nmass, nphi, ntheta, nenergy, nswp])[0,0,:,:,:].squeeze() # [ntheta, nenergy, nswp]
        theta = np.transpose(theta, [1, 0, 2])   # [nenergy, ntheta, nswp]
        dtheta = np.transpose(dtheta, [1, 0, 2]) # [nenergy, ntheta, nswp]

        # energy
        energy  = np.reshape(energy,  [nmass, nphi, ntheta, nenergy, nswp])[0,0,0,:,:].squeeze() # [nenergy, nswp] 
        # denergy = np.reshape(denergy, [nmass, nphi, ntheta, nenergy, nswp])[0,0,0,:,:].squeeze()   # [nenergy, nswp]
        count = np.reshape(count, [len(time), nmass, nphi, ntheta, nenergy])
        dead = np.reshape(dead, [len(time), nmass, nphi, ntheta, nenergy])
        DEF = np.reshape(DEF, [len(time), nmass, nphi, ntheta, nenergy])
        count =  np.transpose(count, axes=(0, 4, 2, 3, 1))
        dead =  np.transpose(dead, axes=(0, 4, 2, 3, 1))
        DEF =  np.transpose(DEF, axes=(0, 4, 2, 3, 1))


        H_DEF = np.squeeze(DEF[:,:,:,:,0])
        O_DEF = np.squeeze(DEF[:,:,:,:,4])
        O2_DEF = np.squeeze(DEF[:,:,:,:,5])
        CO2_DEF = np.squeeze(DEF[:,:,:,:,6])
        
        # Allocate output arrays
        temp_energy = np.zeros((len(time), nenergy))
        # temp_denergy = np.zeros((len(time), nenergy))
        temp_dtheta = np.zeros((len(time), nenergy, ntheta))
        temp_theta = np.zeros((len(time), nenergy, ntheta))
        for i in range(len(time)):
            swp_idx = swp_ind[i]
            temp_energy[i, :] = energy[:, swp_idx]
            # temp_denergy[i, :] = denergy[:, swp_idx]
            temp_theta[i, :] = theta[:, :, swp_idx]
            temp_dtheta[i, :] = dtheta[:, :, swp_idx]
        
        # === Coordinate transformation: quaternion to matrix ===
        sta2mso = np.zeros((len(time), 3, 3))
        q1, q2, q3, q4 = quat_mso[:, 0], quat_mso[:, 1], quat_mso[:, 2], quat_mso[:, 3]
        sta2mso[:, 0, 0] = 1 - 2 * (q3**2 + q4**2)
        sta2mso[:, 0, 1] = 2 * (q2 * q3 - q1 * q4)
        sta2mso[:, 0, 2] = 2 * (q2 * q4 + q1 * q3)
        sta2mso[:, 1, 0] = 2 * (q2 * q3 + q1 * q4)
        sta2mso[:, 1, 1] = 1 - 2 * (q2**2 + q4**2)
        sta2mso[:, 1, 2] = 2 * (q3 * q4 - q1 * q2)
        sta2mso[:, 2, 0] = 2 * (q2 * q4 - q1 * q3)
        sta2mso[:, 2, 1] = 2 * (q3 * q4 + q1 * q2)
        sta2mso[:, 2, 2] = 1 - 2 * (q2**2 + q3**2)
        
        # Store results (flipped energy axis)
        res['time'] = time
        res['energy'] = np.flip(temp_energy, 1)

        res['H_DEF'] = np.flip(H_DEF, 1)
        res['O_DEF'] = np.flip(O_DEF, 1)
        res['O2_DEF'] = np.flip(O2_DEF, 1)
        res['CO2_DEF'] = np.flip(CO2_DEF, 1)

        res['count'] = np.flip(count, 1)
        res['dead'] = np.flip(dead, 1)
        res['dtheta'] = np.flip(temp_dtheta, 1)
        res['theta'] = np.flip(temp_theta, 1)
        res['Bdmpa'] = Bdmpa
        res['scpot'] = scpot
        res['phi'] = phi
        res["sta2mso"] = sta2mso
        # res['dphi'] = dphi
        
    except Exception as e:
        print(f"Warning: Failed to read {filename_d1}. Reason: {e}")

    return res


# %% STATIC d1 mode reader
def read_d1_iv4(filename_d1_iv4):
    """
    Read and process MAVEN STATIC D1 background data from a CDF file.

    Parameters
    ----------
    filename_d1 : str
        Full path to the STATIC D1 iv4 CDF file.

    Returns
    -------
    res : dict
        Dictionary containing processed STATIC D1 data with the following fields:
            - time       [num_time]                            : Time array in datetime64[ns]
            - energy     [num_time, num_energy]                : Energy table (sorted high -> low)
            - denergy    [num_time, num_energy]                : Energy bin width
            - H_DEF      [num_time, num_energy, num_phi, num_theta] : H+ differential energy flux
            - O_DEF      [num_time, num_energy, num_phi, num_theta] : O+ differential energy flux
            - O2_DEF     [num_time, num_energy, num_phi, num_theta] : O2+ differential energy flux
            - CO2_DEF    [num_time, num_energy, num_phi, num_theta] : CO2+ differential energy flux
            - theta      [num_time, num_energy, num_theta]     : Elevation angles
            - dtheta     [num_time, num_energy, num_theta]     : Elevation angle bin width
            - phi        [num_phi]                             : Azimuthal angles [0, 360)
            - dphi       [scalar]                              : Azimuthal bin width
            - Bdmpa      [num_time, 3]                         : Magnetic field in STATIC DMPA frame
            - scpot      [num_time]                            : Spacecraft potential
            - mass       [num_mass]                            : Mass table
    """

    # Define expected dimensions based on STATIC D1 configuration
    nenergy, nmass, nphi, ntheta, nswp = 32, 8, 16, 4, 28
    res = {
        "time": np.array([], dtype="datetime64[ns]"),
        "bkg": np.empty((0, nenergy, nmass), dtype=float),
        "time_integ": np.array([], dtype=float),
        "gf": np.empty((0, nenergy, nmass), dtype=float),
    }

    if not os.path.isfile(filename_d1_iv4):
        print(f"[read_d1_iv4] File not found: {filename_d1_iv4}")
        return res

    try:
        var_names = [
            "epoch", "bkg", "eff", "gf",
            "eff_ind", "att_ind", "swp_ind",
            "geom_factor", "time_integ",
        ]
        is_time = [1] + [0] * (len(var_names) - 1)

        (time, bkg, eff, gf_table,
         eff_ind, att_ind, swp_ind,
         geom_factor, time_integ) = get_cdf_var(
            cdf_filename=filename_d1_iv4,
            variable_name=var_names,
            istime=is_time,
        )

        ntime = len(time)
        eff_ind = eff_ind.astype(int)
        att_ind = att_ind.astype(int)
        swp_ind = swp_ind.astype(int)

 # denergy = np.reshape(denergy, [nmass, nphi, ntheta, nenergy, nswp])[0,0,0,:,:].squeeze()   # [nenergy, nswp]
        bkg = np.reshape(bkg, [len(time), nmass, nphi, ntheta, nenergy])
        bkg = np.transpose(bkg, axes=(0, 4, 2, 3, 1))
        ntime = len(time)
        temp_gf = np.empty((ntime, nenergy, 64), dtype=float);
        temp_eff = np.empty((ntime, nenergy, 64, nmass), dtype=float);
        for i in range(ntime):
            temp_gf[i, :, :]  = np.transpose(np.squeeze(gf_table[att_ind[i], :, :, swp_ind[i]]))  # (ntime, nenergy)
            temp_eff[i, :, :] = np.transpose(np.squeeze(eff[:, :, :, eff_ind[i]]), [2, 1, 0])              # (ntime, nenergy, nmass)
        # Broadcast gf along mass, then multiply by scalar geom_factor
        gf_4d  = temp_gf[:, :, :,  None]              # (ntime, nenergy, ndirection 1)
        gf_eff = geom_factor * gf_4d * temp_eff   # scalar * (ntime, nenergy, ndirection, nmass)

        #
        gf_eff = gf_eff.reshape(ntime, nenergy, 16, 4, nmass)
        bkg = bkg.reshape(ntime, nenergy, 16, 4, nmass)

        # Store results (flipped energy axis)
        gf_eff = np.flip(gf_eff, axis=1)
        bkg = np.flip(bkg, axis=1)
        res["time"] = time
        res["bkg"] = bkg
        res["time_integ"] = time_integ
        res["gf"] = gf_eff

    except Exception as e:
        print(f"Warning: Failed to read {filename_d1_iv4}. Reason: {e}")

    return res

#%% Example usage
if __name__ == "__main__":
    # filename_c0 = 'F:\\data\\maven\\data\\sci\\sta\\l2\\2017\\02\\mvn_sta_l2_c0-64e2m_20170210_v02_r01.cdf'
    # data_c0 = read_c0(filename_c0)

    #filename_c6 = 'F:\\data\\maven\\data\\sci\\sta\\l2\\2017\\02\\mvn_sta_l2_c6-32e64m_20170228_v02.cdf'
    #data_c6 = read_c6(filename_c6)

    # filename_d1 = 'F:\\data\\maven\\data\\sci\\sta\\l2\\2017\\02\\mvn_sta_l2_d1-32e4d16a8m_20170207_v02.cdf'
    # data_d1 = read_d1(filename_d1)

    # filename_c6_iv4 = "F:\\data\\maven\\data\\sci\\sta\\iv4\\2018\\04\\mvn_sta_l2_c6-32e64m_20180401_iv4.cdf"
    # data = read_c6_iv4(filename_c6_iv4)
    filename_d1_iv4 = 'F:\\data\\maven\\data\\sci\\sta\\iv4\\2018\\07\\mvn_sta_l2_d1-32e4d16a8m_20180718_iv4.cdf'
    data = read_d1_iv4(filename_d1_iv4)

 

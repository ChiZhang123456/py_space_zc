import copy
import py_space_zc
import numpy as np  # Importing NumPy for numerical operations
from pyrfu.pyrf import extend_tint  # Importing the 'extend_tint' function from pyrfu.pyrf for time interval extension
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting

def correct_bkg_c6(static_c6):
    """
    Corrects the background for the MAVEN STATIC C6 data, updating the differential energy flux (DEF)
    and removing the background component from the count.

    Parameters:
    -----------
    static_c6 : dict
        A dictionary containing MAVEN STATIC C6 data with keys like 'time', 'count', 'dead', etc.
        - 'time' should be a time array
        - 'count' should be the raw count data
        - 'dead' should be the dead-time correction factor
        - 'DEF' will be calculated and updated here

    Returns:
    --------
    static_c6 : dict
        The updated dictionary containing the corrected background and DEF values.
    """
    # Extend the time interval around the start and end of the data by 10 seconds before and after
    tint = extend_tint([static_c6["time"][0], static_c6["time"][-1]], [-10, 10])

    # Get the background data for the extended time interval
    c6_bkg = py_space_zc.maven.get_data(tint, 'static_c6_iv4')

    # Resample parameters
    time_integ = py_space_zc.resample_time(c6_bkg["time"], c6_bkg['time_integ'], static_c6["time"],
                                            method="linear")
    bkg = py_space_zc.resample_time(c6_bkg["time"], c6_bkg['bkg'], static_c6["time"],
                                            method="linear")
    gf = py_space_zc.resample_time(c6_bkg["time"], c6_bkg['gf'], static_c6["time"],
                                            method="linear")

    # Calculate the new differential energy flux (DEF)
    # Formula: (raw count - background) * dead-time correction / (time integration * geometric factor)
    eflux_new = (static_c6["count"] - bkg) * static_c6["dead"] / (
                time_integ[:, None, None] * gf)

    # Set any negative flux values to NaN (invalid data)
    eflux_new[eflux_new < 0] = np.nan


    new_c6 = copy.deepcopy(static_c6)

    # Update the 'DEF' in the static_c6 dictionary with the newly calculated flux
    new_c6["DEF"] = np.array(eflux_new, copy=True)

    # Update the 'count' field by removing the background component
    new_c6["count"] = np.array(new_c6["count"] - bkg, copy=True)

    # Remove the dead-time correction since it's no longer needed after applying it to the flux
    if "dead" in new_c6:
        del new_c6["dead"]

    return new_c6  # Return the updated dictionary with corrected data



def correct_bkg_d1(static_d1):
    """
    Corrects the background for the MAVEN STATIC D1 data, updating the differential energy flux (DEF)
    and removing the background component from the count.

    Parameters:
    -----------
    static_d1 : dict
        A dictionary containing MAVEN STATIC D1 data with keys like 'time', 'count', 'dead', etc.
        - 'time' should be a time array (e.g., time in seconds or any other consistent unit)
        - 'count' should be the raw count data (typically representing particle counts)
        - 'dead' should be the dead-time correction factor
        - 'DEF' will be calculated and updated here

    Returns:
    --------
    static_d1 : dict
        The updated dictionary containing the corrected background and DEF values.
    """
    # Extend the time interval around the start and end of the data by 10 seconds before and after
    d1_time = static_d1["Bdmpa"].time.data  # Extract the time from the Bdmpa data
    tint = extend_tint([d1_time[0], d1_time[-1]], [-10, 10])  # Define the extended time interval

    # Get the background data for the extended time interval
    d1_bkg = py_space_zc.maven.get_data(tint, 'static_d1_iv4')  # Fetch background data for the given interval

    # Resample parameters to match the time of the static_d1 data
    time_integ = py_space_zc.resample_time(d1_bkg["time"], d1_bkg['time_integ'], d1_time, method="linear")
    bkg = py_space_zc.resample_time(d1_bkg["time"], d1_bkg['bkg'], d1_time, method="linear")
    gf = py_space_zc.resample_time(d1_bkg["time"], d1_bkg['gf'], d1_time, method="linear")

    # Calculate the new differential energy flux (DEF)
    # Formula: (raw count - background) * dead-time correction / (time integration * geometric factor)
    eflux_new = (static_d1["count"] - bkg) * static_d1["dead"] / (time_integ[:, None, None, None, None] * gf)

    # Set any negative flux values to NaN (invalid data)
    eflux_new[eflux_new < 0] = np.nan

    # Extract flux for specific ion species (H, O, O2, CO2) based on the indices
    H_eflux = np.squeeze(eflux_new[:,:,:,:,0])  # Hydrogen (index 0)
    O_eflux = np.squeeze(eflux_new[:,:,:,:,4])  # Oxygen (index 4)
    O2_eflux = np.squeeze(eflux_new[:,:,:,:,5])  # O2 (index 5)
    CO2_eflux = np.squeeze(eflux_new[:,:,:,:,6])  # CO2 (index 6)

    # Store the calculated DEF values for different species
    new_d1 = copy.deepcopy(static_d1)
    new_d1["H_DEF"].data.data = H_eflux.copy()
    new_d1["O_DEF"].data.data = O_eflux.copy()
    new_d1["O2_DEF"].data.data = O2_eflux.copy()
    new_d1["CO2_DEF"].data.data = CO2_eflux.copy()

    # Clean up by deleting unnecessary keys from the static_d1 dictionary
    del new_d1["dead"]
    del new_d1["count"]

    return new_d1  # Return the updated dictionary with corrected data

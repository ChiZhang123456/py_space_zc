import numpy as np
import pandas as pd


def read_gwen_density_txt(filename):
    """
    Reads the Gwen density data from a specified text file and returns the time series and density values
    for several species (H, He, O, O2).

    Parameters:
    -----------
    filename : str
        The path to the text file containing the Gwen density data. The file should have columns like:
        'Time', 'nH', 'nHe', 'nO', 'nO2', where 'Time' is a timestamp and the others are the corresponding densities.

    Returns:
    --------
    res : dict
        A dictionary containing the following:
        - 'time' : A list of `numpy.datetime64` objects representing the time series.
        - 'nH'   : A NumPy array of hydrogen (H) densities.
        - 'nHe'  : A NumPy array of helium (He) densities.
        - 'nO'   : A NumPy array of oxygen (O) densities.
        - 'nO2'  : A NumPy array of oxygen (O2) densities.

    Example:
    --------
    res = read_gwen_density_txt('gwen_density_data.txt')
    print(res['time'])  # Prints the list of timestamps
    print(res['nH'])    # Prints the hydrogen density values
    """

    # Read the data from the text file using pandas. The separator is one or more spaces ('\s+')
    df = pd.read_csv(filename, sep='\s+')

    # Replace the slash (/) in the 'Time' column with a space to make it compatible with datetime conversion
    df['Time'] = df['Time'].str.replace('/', ' ')

    # Convert the 'Time' column into numpy.datetime64 objects for time-series processing
    time = np.array([np.datetime64(Time) for Time in df['Time']])

    # Extract the species density values as numpy arrays for further processing
    nH = df['nH'].values  # Hydrogen density
    nHe = df['nHe'].values  # Helium density
    nO = df['nO'].values  # Oxygen density
    nO2 = df['nO2'].values  # Oxygen 2 (O2) density

    # Create a dictionary to hold the time and species data
    res = {'time': time, 'nH': nH, 'nHe': nHe, 'nO': nO, 'nO2': nO2}

    # Return the dictionary containing the processed data
    return res

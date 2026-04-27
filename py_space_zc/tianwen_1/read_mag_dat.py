import pandas as pd
import numpy as np


def read_mag_dat_file(filename):
    """
    Read Tianwen-1 MOMAG .dat file.

    Parameters:
        filename (str): Path to the .dat file.

    Returns:
        time (np.ndarray): 1D array of datetime64[ns].
        Bmso (np.ndarray): Nx3 array of Magnetic Field components (nT).
        Pmso (np.ndarray): Nx3 array of Spacecraft Position (km).
    """
    # 1. Load data using pandas
    # comment='#' ignores metadata lines starting with ##
    # sep='\s+' handles multiple spaces between columns
    df = pd.read_csv(filename,
                     comment='#',
                     sep='\s+',
                     header=None,
                     usecols=[0, 2, 3, 4, 5, 6, 7],
                     names=['time', 'Bx', 'By', 'Bz', 'Px', 'Py', 'Pz'],
                     engine='python')

    # 2. Convert ISO 8601 strings to numpy datetime64[ns]
    # .values returns the underlying numpy array
    time = pd.to_datetime(df['time']).values.astype('datetime64[ns]')

    # 3. Extract MSO coordinates as Nx3 numpy arrays
    Bmso = df[['Bx', 'By', 'Bz']].values
    Pmso = df[['Px', 'Py', 'Pz']].values

    return time, Bmso, Pmso


# --- Main Execution ---
if __name__ == "__main__":
    # Use 'r' before the string to handle Windows backslashes
    file_path = r"F:\Tianwen_1\MAG\2025\TW1_MOMAG_MSO_32Hz_20250615_2C_v03.dat"

    try:
        time, Bmso, Pmso = read_mag_dat_file(file_path)

        print("File read successfully.")
        print(f"Total data points: {len(time)}")
        print(f"First timestamp:   {time[0]}")
        print(f"First B vector:    {Bmso[0, :]} nT")
        print(f"First P vector:    {Pmso[0, :]} km")

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
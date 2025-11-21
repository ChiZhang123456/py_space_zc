import pandas as pd
import numpy as np

def read_els_pad(filename):
    """
    Read VEX ELS PAD data and convert it into a 3D array of
    [time, energy, pitch angle].

    Parameters
    ----------
    filename : str
        Path to the CSV file, typically with header information in the first 3 rows,
        and comma-separated numeric data starting from the 4th row.

    Returns
    -------
    time : ndarray of numpy.datetime64
        Unique time stamps (start times) for each measurement interval.

    energy_matrix : ndarray of shape (N_time, N_energy)
        Energy value for each [time, energy] grid point (in eV).

    pa : ndarray of shape (N_pitch_angle,)
        Pitch angle center values in degrees (5°–175°).

    pad : ndarray of shape (N_time, N_energy, N_pitch_angle)
        PAD values (pitch angle distributions) for each time-energy-angle combination.
        Invalid values (e.g., |value| > 1.0) are set to NaN.
    """

    # Read CSV file, skip the first 3 header lines
    df = pd.read_csv(filename, skiprows=3, header=None, sep=',')

    # Convert to NumPy array for fast slicing
    data = df.to_numpy()

    # Extract relevant columns
    start_time = data[:, 0]                # String timestamps
    energy = data[:, 3].astype(float)      # Energy in eV
    pad = data[:, 5:].astype(float)        # PAD values (shape: [N_total_row, 18])

    # Mask invalid values: absolute PAD > 1.0 → set to NaN
    pad[np.abs(pad) > 1.0] = np.nan

    # Convert time strings to datetime format
    time_dt = pd.to_datetime(start_time, format='%Y-%jT%H:%M:%S.%f')

    # Get all unique time points
    time_unique = sorted(time_dt.unique())
    N_time = len(time_unique)

    # Count how many energy levels per time (should be consistent)
    counts = time_dt.value_counts().sort_index()
    assert counts.nunique() == 1, "Number of energy levels per time is not constant."

    N_energy = counts.iloc[0]
    N_pad = pad.shape[1]  # usually 18 pitch angles

    # Reshape to [time, energy, PAD]
    pad_3d = pad.reshape((N_time, N_energy, N_pad))
    energy_matrix = energy.reshape((N_time, N_energy))

    # Pitch angle centers
    pa = np.linspace(5, 175, N_pad)

    # Convert time to numpy.datetime64 array
    time_np = np.array(time_unique, dtype='datetime64[ns]')
    energy_matrix = np.flip(energy_matrix, axis = 1)
    pad_3d = np.flip(pad_3d, axis = 1)
    return time_np, energy_matrix, pa, pad_3d


if __name__ == '__main__':
    filename = r"C:\Users\Win\Downloads\VExELSPADRG_2006135_Data.csv"
    time, energy_matrix, pa, pad_3d = read_els_pad(filename)

    print("Time shape:", time.shape)
    print("Energy shape:", energy_matrix.shape)
    print("PAD shape:", pad_3d.shape)
    print("First time:", time[0])

import numpy as np
import xarray as xr
import os
from py_space_zc import get_cdf_var


def read_resample_pad(filename, tint=None):
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return None

    try:
        # === Step 1: Read variables ===
        val_names = ['epoch', 'avg', 'xax', 'std']
        is_time = [1, 0, 0, 0]
        time, avg, xax, std = get_cdf_var(cdf_filename=filename, variable_name=val_names, istime=is_time)

        # Remove singleton dimensions
        avg, xax, std = avg.squeeze(), xax.squeeze(), std.squeeze()

        # === Step 2: Pitch Angle and Energy setup ===
        pitchangle = xax[0, :]

        # Transpose to [Time, Energy, PitchAngle]
        avg = np.transpose(avg, (0, 2, 1))

        energy = np.array([
            4627.5, 4118.506, 3665.5, 3262.32, 2903.48, 2584.12, 2299.88, 2046.91,
            1821.76, 1621.38, 1443.04, 1284.317, 1143.05, 1017.32, 905.42, 805.83,
            717.2, 638.31, 568.1, 505.6, 449.9989, 400.5, 356.45, 317.242, 282.35,
            251.29, 223.65, 199.051, 177.157, 157.671, 140.328, 124.893, 111.155,
            98.929, 88.05, 78.363, 69.74, 62.1, 55.24, 49.17, 43.76, 38.95, 34.666,
            30.85, 27.457, 24.437, 21.749, 19.357, 17.23, 15.333, 13.646, 12.145,
            10.81, 9.62, 8.56, 7.62, 6.78, 6.036, 5.37, 4.78, 4.255, 3.79, 3.37, 3.0
        ])

        # === Step 3: Time Slicing ===
        if tint is not None:
            # Ensure tint[0] and tint[1] are converted to numpy datetime64
            t_start = np.datetime64(tint[0])
            t_stop = np.datetime64(tint[1])
            mask = (time >= t_start) & (time <= t_stop)
            time = time[mask]
            avg = avg[mask, :, :]
            std = std[mask, :, :]

        # === Step 4: Flip Data (Energy Axis) ===
        # Flip energy list and the energy dimension (axis 1) of data
        energy = np.flip(energy, axis=0)
        avg = np.flip(avg, axis=1)

        # === Step 5: Construct Xarray Dataset ===
        # Use coords for independent variables and data_vars for the measurements
        pad = xr.Dataset(
            data_vars={
                "data": (["time", "energy", "pitchangle"], avg),
            },
            coords={
                "time": time,
                "energy": energy,
                "pitchangle": pitchangle,
            }
        )

        # Add Metadata
        pad.attrs = {
            'species': 'e-',
            "units": 'keV/(cm^2 s sr keV)',
            "description": "Pitch Angle Distribution (PAD)"
        }
        pad.data.attrs["units"] = 'keV/(cm^2 s sr keV)'

        return pad

    except Exception as e:
        print(f"Error processing CDF: {e}")
        return None


if __name__ == "__main__":
    # Example path - ensure it uses raw string r'' for Windows paths
    filename = r'D:\Work_Work\Mars\MAVEN\HCS_to_FB\20250615_113000_20250615_123000.cdf'
    # Define a time interval to test slicing
    tint = ["2025-06-15T11:40:00", "2025-06-15T12:00:00"]
    res = read_resample_pad(filename, tint=tint)
"""
This script demonstrates how to transform the MSO (Mars–Solar–Orbital)
coordinate system basis vectors into the MAVEN STATIC instrument coordinate
system (STA frame) using MAVEN SPICE kernels.

IMPORTANT:
----------
Before calling `get_mso_sta`, you must first load the required MAVEN SPICE kernels
by calling `maven.load_maven_spice()`. This ensures all frame definitions, SCLK,
LSK, and CK kernels are available for coordinate transformation.


Functions:
----------
get_mso_sta(time):
    Computes the MSO basis vectors (X, Y, Z) at specified times, then transforms
    them into the MAVEN STATIC coordinate system (STA).

Example:
--------
if __name__ == "__main__":
    start_time = np.datetime64("2020-10-27T11:39")
    end_time   = np.datetime64("2020-10-27T11:45")
    time       = time_linspace(start_time, end_time, number=20)
    maven.load_maven_spice()
    xsta, ysta, zsta = get_mso_sta(time)
    sp.kclear()  # Clear loaded SPICE kernels
"""

from py_space_zc import maven, ts_vec_xyz, time_linspace, resample_time
import numpy as np
import spiceypy as sp

def get_mso_sta_via_spice(time):
    """
    Transform MSO basis vectors (X, Y, Z) into MAVEN STATIC coordinates.

    Parameters
    ----------
    time : array-like of numpy.datetime64
        Array of time points for which to perform the transformation.

    Returns
    -------
    xsta, ysta, zsta : object
        The X, Y, Z unit vectors of the MSO frame expressed in the STATIC frame.
        These are returned as time series objects from `ts_vec_xyz`.

    Notes
    -----
    - This function first constructs three orthogonal unit vectors in MSO
      coordinates:
        xmso = (1, 0, 0)
        ymso = (0, 1, 0)
        zmso = (0, 0, 1)
      at each given time.
    - These unit vectors are then wrapped in `ts_vec_xyz` objects so they are
      compatible with the `maven.coords_convert` function.
    - Finally, each vector is transformed from MSO to STA coordinates.

    - You must load MAVEN SPICE kernels via `maven.load_maven_spice()` before
      calling this function, otherwise the coordinate conversion will fail.
    """

    ntime = len(time)                     # Number of time points

    # Create arrays for the three MSO unit vectors
    xmso = np.zeros([ntime, 3])
    ymso = np.zeros([ntime, 3])
    zmso = np.zeros([ntime, 3])

    # Assign 1 to the appropriate component for each unit vector
    xmso[:, 0] = 1   # X-axis
    ymso[:, 1] = 1   # Y-axis
    zmso[:, 2] = 1   # Z-axis

    # Convert NumPy arrays to time series vector format (required by coords_convert)
    xmso = ts_vec_xyz(time, xmso)
    ymso = ts_vec_xyz(time, ymso)
    zmso = ts_vec_xyz(time, zmso)

    # Transform from MSO to STA coordinates
    xsta = maven.coords_convert(xmso, "mso2sta")
    ysta = maven.coords_convert(ymso, "mso2sta")
    zsta = maven.coords_convert(zmso, "mso2sta")
    return xsta, ysta, zsta







if __name__ == "__main__":
    # Define start and end times
    start_time = np.datetime64("2020-10-27T11:39")
    end_time   = np.datetime64("2020-10-27T11:45")

    # Generate 20 equally spaced time points between start_time and end_time
    time = time_linspace(start_time, end_time, number=20)

    # Load required MAVEN SPICE kernels (must be done before coordinate transforms)
    maven.load_maven_spice()

    # Perform the MSO → STA coordinate transformation
    xsta, ysta, zsta = get_mso_sta_via_spice(time)

    # Unload all SPICE kernels from memory
    sp.kclear()

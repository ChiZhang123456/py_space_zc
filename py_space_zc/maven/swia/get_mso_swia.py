import numpy as np
import spiceypy as sp
from py_space_zc import maven, ts_vec_xyz

def get_mso_swia(time):
    """
    Convert unit vectors of MSO (Mars-Solar-Orbital) coordinates into the
    SWIA (Solar Wind Ion Analyzer) instrument frame for a given time series.

    This function is primarily used for transforming the three Cartesian unit vectors
    (x̂, ŷ, ẑ in MSO) into the SWIA instrument frame using SPICE kernels.

    Parameters
    ----------
    time : array-like of datetime64
        Array of time values for which the transformation is computed.

    Returns
    -------
    xmso_swia : xarray.DataArray
        Time series of the MSO-x unit vector transformed into SWIA frame.
    ymso_swia : xarray.DataArray
        Time series of the MSO-y unit vector transformed into SWIA frame.
    zmso_swia : xarray.DataArray
        Time series of the MSO-z unit vector transformed into SWIA frame.

    Notes
    -----
    - This function clears the SPICE kernel pool at both beginning and end using `sp.kclear()`.
    - It uses the `py_space_zc.maven.coords_convert` function for frame transformation.
    - Requires SPICE kernels to be correctly loaded via `maven.load_maven_spice()`.

    Example
    -------
    >>> tint = ["2022-01-24T08:06:30", "2022-01-24T08:09:00"]
    >>> swia_3d = maven.load_data(tint, 'swia_3d')
    >>> xmso_swia, ymso_swia, zmso_swia = get_mso_swia(swia_3d.time.data)
    """
    sp.kclear()  # Clear existing SPICE kernels to avoid contamination
    maven.load_maven_spice()  # Load MAVEN SPICE kernels

    n_t = len(time)  # Number of time steps

    # Initialize MSO unit vectors (constant in MSO frame)
    xmso = np.zeros((n_t, 3), dtype=np.float64)
    ymso = np.zeros((n_t, 3), dtype=np.float64)
    zmso = np.zeros((n_t, 3), dtype=np.float64)
    xmso[:, 0] = 1.0  # x-direction
    ymso[:, 1] = 1.0  # y-direction
    zmso[:, 2] = 1.0  # z-direction

    # Wrap them into time series vector format
    xmso_ts = ts_vec_xyz(time, xmso)
    ymso_ts = ts_vec_xyz(time, ymso)
    zmso_ts = ts_vec_xyz(time, zmso)

    # Transform from MSO to SWIA instrument frame
    xmso_swia = maven.coords_convert(xmso_ts, 'mso2swia')
    ymso_swia = maven.coords_convert(ymso_ts, 'mso2swia')
    zmso_swia = maven.coords_convert(zmso_ts, 'mso2swia')

    sp.kclear()  # Clean up SPICE pool again after use

    return xmso_swia, ymso_swia, zmso_swia


if __name__ == '__main__':
    # Define time interval for the data to be processed
    tint =  ["2022-01-24T08:06:30", "2022-01-24T08:09:00"]

    # Load magnetic field and SWIA data using py_space_zc MAVEN wrapper
    swia_3d = maven.load_data(tint, 'swia_3d')

    # Convert MSO unit vectors into SWIA frame over the SWIA time array
    xmso_swia, ymso_swia, zmso_swia = get_mso_swia(swia_3d.time.data)

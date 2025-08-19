"""
Retrieve the MAVEN MSO basis vectors expressed in the STATIC coordinate frame
from pre-loaded STATIC D1 mode data.

Description
-----------
This script extracts the MSO coordinate system unit vectors (X, Y, Z)
as represented in the MAVEN STATIC instrument's frame from STATIC D1 mode data.
The transformation matrix from STA → MSO is stored in the 'sta2mso' variable
within the D1 dataset.

Usage Notes
-----------
- The input `d1` must be a dictionary-like object returned by
  `py_space_zc.maven.get_data()` with the variable 'sta2mso' available.
- 'sta2mso' is expected to have shape: [time, 3, 3]
    - The second dimension indexes the MSO unit vectors (0=X, 1=Y, 2=Z)
    - The third dimension indexes the vector components in STA coordinates
- This function simply extracts and squeezes the relevant slices
  so each output is an array of shape [time, 3] representing one unit vector.

Dependencies
------------
- NumPy
- py_space_zc (custom MAVEN data handling library)

Example
-------
if __name__ == "__main__":
    start_time = "2020-10-27T11:39"
    end_time   = "2020-10-27T11:45"

    # Retrieve MAVEN STATIC D1 mode data for the given time range
    d1 = py_space_zc.maven.get_data([start_time, end_time], 'static_d1')

    # Extract MSO basis vectors expressed in STA coordinates
    xmso_sta, ymso_sta, zmso_sta = get_mso_sta_via_d1(d1)
"""

import numpy as np
import py_space_zc

def get_mso_sta_via_d1(d1):
    """
    Extract MSO basis vectors (X, Y, Z) in STA coordinates from STATIC D1 data.

    Parameters
    ----------
    d1 : dict
        Dictionary containing MAVEN STATIC D1 data. Must include the key 'sta2mso'
        with shape [time, 3, 3].

    Returns
    -------
    xmso_sta : ndarray, shape (time, 3)
        X-axis of MSO frame expressed in STA coordinates at each time step.
    ymso_sta : ndarray, shape (time, 3)
        Y-axis of MSO frame expressed in STA coordinates.
    zmso_sta : ndarray, shape (time, 3)
        Z-axis of MSO frame expressed in STA coordinates.
    """
    # Extract the first, second, and third MSO unit vectors from 'sta2mso'
    # and remove any singleton dimensions using np.squeeze
    xmso_sta = np.squeeze(d1["sta2mso"][:, 0, :])  # MSO X-axis in STA coords
    ymso_sta = np.squeeze(d1["sta2mso"][:, 1, :])  # MSO Y-axis in STA coords
    zmso_sta = np.squeeze(d1["sta2mso"][:, 2, :])  # MSO Z-axis in STA coords

    return xmso_sta, ymso_sta, zmso_sta


if __name__ == "__main__":
    # Define time range of interest
    start_time = "2020-10-27T11:39"
    end_time   = "2020-10-27T11:45"

    # Retrieve STATIC D1 mode data for this time range
    d1 = py_space_zc.maven.get_data([start_time, end_time], 'static_d1')

    # Extract MSO basis vectors expressed in STA coordinates
    xmso_sta, ymso_sta, zmso_sta = get_mso_sta_via_d1(d1)

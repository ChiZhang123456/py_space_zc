import numpy as np
import py_space_zc

def sta2mso(inp, d1):
    """
    Convert a vector time series from STATIC to MSO coordinates using the
    per-time rotation matrix contained in STATIC D1 data.

    Parameters
    ----------
    inp : TSeries-like (vector, shape = [time, 3])
        Input vector in STATIC coordinates. Must be a pyrfu/py_space_zc time series
        with `.time` and `.data` fields.
    d1 : dict
        STATIC D1 dataset returned by py_space_zc.maven.get_data(...).
        Must contain:
          - 'time'    : array-like of times (aligned target grid)
          - 'sta2mso' : ndarray of shape [ntime, 3, 3], rotation from STA -> MSO

    Returns
    -------
    out : TSeries-like
        The same vector expressed in MSO coordinates, on d1['time'] grid.
        out.attrs['Coordinates'] == 'MSO'

    Notes
    -----
    - If R_sta2mso(t) maps STA → MSO, then MSO → STA is R_mso2sta(t) = R_sta2mso(t).T
    - We resample the input to the D1 time grid, then apply a batched matrix multiply.

    Example
    -------
    >>> d1 = py_space_zc.maven.get_data(["2020-10-27T11:39","2020-10-27T11:45"], "static_d1")
    >>> v_mso = py_space_zc.ts_vec_xyz(d1["time"], some_vector_in_mso)  # shape [N,3]
    >>> v_sta = mso2sta(v_mso, d1)
    """
    # Resample input vector to the D1 time grid (target grid)
    t_target = d1["time"]
    inp_resampled = py_space_zc.resample_time(inp.time.data, inp.data, t_target, 'linear')

    # Extract data and rotation matrices
    R_sta2mso = d1["sta2mso"]             # shape: [N, 3, 3]

    # Batched matrix multiply: for each time step i, v_sta[i] = R_mso2sta[i] @ v_mso[i]
    v_mso = np.einsum('nij,nj->ni', R_sta2mso, inp_resampled)  # shape: [N, 3]

    # Wrap back to a vector time series with STATIC coord tag
    out = py_space_zc.ts_vec_xyz(t_target, v_mso,
                                 attrs={'Coordinates': 'MSO'})
    return out


if __name__ == "__main__":
    # Define time range of interest
    start_time = "2020-10-27T11:39"
    end_time   = "2020-10-27T11:45"

    # Retrieve STATIC D1 mode data for this time range
    d1 = py_space_zc.maven.get_data([start_time, end_time], 'static_d1')

    # Extract MSO basis vectors expressed in STA coordinates
    xmso = np.zeros((len(d1["time"]), 3))
    ymso = np.zeros((len(d1["time"]), 3))
    zmso = np.zeros((len(d1["time"]), 3))
    xmso[:, 0] = 1.0
    ymso[:, 1] = 1.0
    zmso[:, 2] = 1.0
    xmso = py_space_zc.ts_vec_xyz(d1["time"], xmso)
    ymso = py_space_zc.ts_vec_xyz(d1["time"], ymso)
    zmso = py_space_zc.ts_vec_xyz(d1["time"], zmso)
    xmso_sta = py_space_zc.maven.static.mso2sta(xmso, d1)
    ymso_sta = py_space_zc.maven.static.mso2sta(ymso, d1)
    zmso_sta = py_space_zc.maven.static.mso2sta(zmso, d1)
    xmso_new = sta2mso(xmso_sta,d1)
    ymso_new = sta2mso(ymso_sta,d1)
    zmso_new = sta2mso(zmso_sta,d1)
    print(xmso_new)
    print(ymso_new)
    print(zmso_new)




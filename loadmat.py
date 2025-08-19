import scipy.io as sio
import h5py
import numpy as np


def loadmat(filename):
    """
    Load MATLAB .mat files (both v7.3 and earlier).

    Parameters
    ----------
    filename : str
        Path to the .mat file.

    Returns
    -------
    dict
        Dictionary containing variables from the .mat file.
        Arrays are returned as NumPy arrays with MATLAB-style orientation.
    """
    try:
        # Try using scipy for v7.2 and earlier
        return sio.loadmat(filename)
    except NotImplementedError:
        # v7.3 files -> use h5py
        data_dict = {}
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                arr = np.array(f[key])
                # Transpose multi-dimensional arrays to match MATLAB layout
                if arr.ndim > 1:
                    arr = arr.T
                data_dict[key] = arr
        return data_dict

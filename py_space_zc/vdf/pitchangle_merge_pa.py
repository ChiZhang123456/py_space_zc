import numpy as np
from py_space_zc import ts_spectr

def pitchangle_merge_pa(dataset_pad, parange: list):
    """
    Extract a reduced energy-time spectrogram by averaging the PAD (pitch angle distribution)
    data over a specified pitch angle range.

    Parameters
    ----------
    dataset_pad : xr.Dataset
        Input xarray dataset containing pitch angle-resolved data. Must include:
        - time.data       : array-like of shape (ntime,)
                            Time stamps for each data sample.
        - data            : ndarray of shape (ntime, nenergy, npitch)
                            Data cube, such as PSD or DEF, organized by time, energy, and pitch angle.
        - energy.data     : ndarray
                            Energy bins, either static (shape: (nenergy,))
                            or time-varying (shape: (ntime, nenergy)).
        - pitchangle.data : array-like of shape (npitch,)
                            Pitch angle bin centers (in degrees).
        - attrs           : dict-like
                            Metadata dictionary to be included in the output spectrogram.

    parange : list or tuple of float
        Two-element range [theta_min, theta_max] (in degrees) over which the pitch angle
        data should be averaged or integrated.

    Returns
    -------
    ts_spectr
        A spectrogram object containing:
        - time   : array-like, time axis (same as input)
        - energy : array-like, energy bins
        - data   : ndarray of shape (ntime, nenergy), pitch angle–averaged values
    """

    # Extract variables from the dataset
    time = dataset_pad.time.data
    energy = dataset_pad.energy.data
    pitchangle = dataset_pad.pitchangle.data
    data = dataset_pad.data

    # Create a mask to select pitch angle bins within the specified range
    pa_mask = (pitchangle >= parange[0]) & (pitchangle <= parange[1])

    # Select data within pitch angle range → shape: (ntime, nenergy, selected_pitch)
    pad_selected = data[:, :, pa_mask]

    # Average over selected pitch angles → shape: (ntime, nenergy)
    pad_reduced = np.nansum(pad_selected, axis=2)

    # Return as ts_spectr spectrogram object with inherited metadata
    result = ts_spectr(
        time,
        energy,
        pad_reduced,
        attrs=dataset_pad.attrs,
    )
    return result

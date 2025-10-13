import numpy as np
from py_space_zc import ts_spectr

def pitchangle_merge_energy(dataset_pad, energyrange: list, option=None):
    """
    Compute a pitch angle distribution (PAD) time series by averaging input
    data over a specified energy range.

    Parameters
    ----------
    dataset_pad : xr.Dataset
        Input dataset containing:
        - time.data       : array-like, shape (ntime,)
        - data            : ndarray, shape (ntime, nenergy, npitch)
                            Data cube (e.g., DEF or PSD) over time, energy, and pitch angle.
        - energy.data     : ndarray
                            Energy values, either static (shape: (nenergy,))
                            or time-varying (shape: (ntime, nenergy)).
        - pitchangle.data : array-like, shape (npitch,)
                            Pitch angle bins in degrees.
        - attrs           : dict-like
                            Metadata to be passed into the output spectrogram.

    energyrange : list or tuple of float
        Energy range [E_min, E_max] in eV to integrate/average over.

    option : str or None, optional
        If 'norm': normalize PAD by its mean over pitch angles at each time step.
        If None: return absolute PAD values.

    Returns
    -------
    ts_spectr
        Spectrogram object containing:
        - time        : array-like, timestamps
        - pitchangle  : array-like, pitch angle bins
        - data        : ndarray, shape (ntime, npitch)
                        PAD as a function of time and pitch angle.
                        Optionally normalized.
    """

    time = dataset_pad.time.data
    energy = dataset_pad.energy.data
    pitchangle = dataset_pad.pitchangle.data
    data = dataset_pad.data

    # -------- Energy selection --------
    # Case 1: Static energy bins (shape: (nenergy,))
    if energy.ndim == 1:
        energy_mask = (energy >= energyrange[0]) & (energy <= energyrange[1])
        pad_selected = data[:, energy_mask, :]  # shape → (ntime, nselected, npitch)

    # Case 2: Time-varying energy bins (shape: (ntime, nenergy))
    elif energy.ndim == 2:
        pad_selected = []
        for i in range(data.shape[0]):
            e_row = energy[i, :]
            mask = (e_row >= energyrange[0]) & (e_row <= energyrange[1])
            pad_selected.append(data[i, mask, :])
        pad_selected = np.array(pad_selected, dtype=np.float64)  # shape → (ntime, ?, npitch)

    else:
        raise ValueError("`energy` must be either 1D (static) or 2D (time-varying).")

    # -------- Energy averaging --------
    # Average across selected energy bins to get PAD vs. pitch angle
    pad_reduced = np.nanmean(pad_selected, axis=1)  # shape → (ntime, npitch)

    # -------- Optional normalization --------
    if option is None:
        result_data = pad_reduced
    elif option == 'norm':
        mean_pad = np.nanmean(pad_reduced, axis=1)  # average over pitch angles
        mean_pad[mean_pad == 0] = np.nan            # avoid division by zero
        result_data = pad_reduced / mean_pad[:, None]
    else:
        raise ValueError("`option` must be None or 'norm'.")

    # -------- Construct spectrogram --------
    result = ts_spectr(
        time,
        pitchangle,
        result_data,
        comp_name="Pitch Angle",
        attrs=dataset_pad.attrs,
    )
    # result.data.attrs["UNITS"] = dataset_pad.data.attrs["UNITS"]
    # result.time.attrs = dataset_pad.time.attrs

    return result

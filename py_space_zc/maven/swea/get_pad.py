from py_space_zc import maven, vdf
import xarray as xr
import numpy as np

def get_pad(tint, delta_angles=22.5):
    """
    Load and compute pitch angle distribution (PAD) from MAVEN SWEA data.

    Parameters
    ----------
    tint : list or tuple of str
        Time interval in the form ["YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DDTHH:MM:SS"].

    delta_angles : float, optional
        Bin width for pitch angle calculation in degrees.
        Default is 22.5°, yielding 8 bins from 0–180°.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'time':      1D array of time stamps
        - 'energy':    1D array of energy bins [eV]
        - 'pitchangle': 1D array of pitch angle bin centers [degrees]
        - 'data':      3D array of PAD, shape (ntime, nenergy, npitchangle)
    """

    # Load SWEA pitch angle data from MAVEN
    swea_pad = maven.load_data(tint, 'swea_pad')

    # Compute PAD using pitchangle_dis function
    pitchangle, pad_data = vdf.pitchangle_dis(
        data=swea_pad['DEF'],
        pa_dis=swea_pad['PA'],
        delta_angles=delta_angles
    )

    # Organize output
    n_time = len(swea_pad['time'])
    n_energy = len(swea_pad['energy'])
    pad = xr.Dataset(
        {
            "data": (["time", "idx0", "idx1"], pad_data,),
            "energy": (["time", "idx0"], np.tile(swea_pad['energy'], (n_time, 1))),
            "pitchangle": (["idx1"], pitchangle),
            "time": swea_pad['time'],
            "idx0": np.arange(n_energy),
            "idx1": np.arange(len(pitchangle)),
        },)

    pad.attrs = {'species':'e-',
                 "delta_pitchangle_minus": delta_angles * 0.5,
                 "delta_pitchangle_plus": delta_angles * 0.5,
                 "UNITS": 'keV/(cm^2 s sr keV)'}

    pad.data.attrs["UNITS"] = 'keV/(cm^2 s sr keV)'

    return pad

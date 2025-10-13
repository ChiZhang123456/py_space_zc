import numpy as np
from py_space_zc import maven, vdf
from pyrfu.pyrf import normalize, resample
import spiceypy as sp

def get_pad(tint, delta_angles=22.5):
    """
    Compute pitch angle distribution (PAD) from MAVEN SWIA 3D data.

    This function loads the magnetic field and ion distribution data,
    converts the magnetic field to SWIA frame, resamples to match SWIA timestamps,
    and computes the pitch angle distribution in the spacecraft frame.

    Parameters
    ----------
    tint : list of str
        Time interval, e.g., ["2023-05-01T00:00:00", "2023-05-01T00:10:00"]

    delta_angles : float, optional
        Bin width for pitch angle [degrees], default is 22.5°.

    Returns
    -------
    xr.Dataset
        Pitch angle distribution with dimensions (time, energy, pitchangle)
        and coordinates including energy grid, pitch angle bins, etc.

    Notes
    -----
    Requires MAVEN SPICE kernels to be loaded beforehand via:
        maven.load_maven_spice()
    """
    sp.kclear()
    maven.load_maven_spice()

    # Load magnetic field and SWIA 3D distribution
    B, swia_3d = maven.load_data(tint, ['B', 'swia_3d'])

    # Convert magnetic field from MSO to SWIA coordinates
    Bswia = maven.coords_convert(B['Bmso'], 'mso2swia')

    # Resample magnetic field to match SWIA timestamps
    Bswia = resample(Bswia, swia_3d)

    # Normalize B vector to get pitch angle direction
    bswia = normalize(Bswia)

    # Compute PAD (time × energy × pitch angle)
    pad = vdf.pitchangle_dis_3d(swia_3d, bswia, delta_angles=delta_angles)
    sp.kclear()
    return pad

if __name__ == '__main__':
    tint = ["2018-10-18T20:10:00", "2018-10-18T20:30:00"]
    sp.kclear()
    maven.load_maven_spice()
    swia_pad = get_pad(tint)
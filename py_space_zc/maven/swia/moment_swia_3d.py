from py_space_zc import maven, vdf, rotate_tensor
import numpy as np
import spiceypy as sp

def moment_swia_3d(swia_3d):
    """
    Calculate plasma moments from MAVEN SWIA 3D ion distributions,
    and transform velocity-space tensors to MSO coordinates.

    Parameters
    ----------
    swia_3d : xarray.Dataset
        MAVEN SWIA 3D ion distribution data. Must contain DEF or PSD,
        energy table, angular bins, and time dimension.

    Returns
    -------
    moment : dict of xarray.DataArray
        Dictionary of computed moments with keys:
        - 'n'     : Number density        [cm⁻³]
        - 'P'     : Scalar pressure       [nPa]
        - 'Vmso'  : Bulk velocity (MSO)   [m/s]
        - 'Hmso'  : Pressure tensor (MSO) [nPa]
        - 'Kmso'  : Heat flux tensor (MSO) [eV/cm²/s]
        - 'Qmso'  : Heat flux vector (MSO) [W/m²]

    Notes
    -----
    - All moments are computed assuming isotropic mass (default: proton mass).
    - No spacecraft potential or velocity correction is applied (set to None).
    - Outputs are aligned with the original SWIA time resolution.
    - Vector/tensor quantities are returned in Mars Solar Orbital (MSO) coordinates.
    """

    # --- Step 1: Load SPICE kernels (time-dependent coordinate transforms)
    sp.kclear()
    maven.load_maven_spice()
    psd = vdf.flux_convert(swia_3d, 'def2psd')
    
    # --- Step 2: Compute plasma moments in instrument frame
    moment = vdf.moments_calculation(
        psd,
        sc_pot=None,           # no spacecraft potential correction
        vsc_instrument=None    # no spacecraft velocity subtraction
    )

    # --- Step 3: Transform vector/tensor outputs to MSO coordinates
    Vmso = maven.coords_convert(moment['V'], 'swia2mso')   # Bulk velocity
    Hmso = maven.coords_convert(moment['H'], 'swia2mso')   # Pressure tensor
    Kmso = maven.coords_convert(moment['K'], 'swia2mso')   # Heat flux tensor
    Qmso = maven.coords_convert(moment['Q'], 'swia2mso')   # Heat flux vector

    # --- Step 4: Replace original with MSO-transformed quantities
    del moment['V']
    del moment['H']
    del moment['K']
    del moment['Q']

    moment['Vmso'] = Vmso
    moment['Hmso'] = Hmso
    moment['Kmso'] = Kmso
    moment['Qmso'] = Qmso

    return moment


# --------------------------------------------------------------------------------------
# Example usage: Compute moments from MAVEN SWIA 3D data and transform to MSO frame
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Define time interval for data loading
    tint = ["2018-10-18T20:10:00", "2018-10-18T20:30:00"]

    # Load magnetic field and SWIA 3D data
    B, swia_3d = maven.load_data(tint, ['B', 'swia_3d'])

    # Compute plasma moments in MSO frame
    moment = moment_swia_3d(swia_3d)

    # Access results
    print(moment.keys())
    print(moment['n'])         # number density
    print(moment['Vmso'])      # velocity vector in MSO

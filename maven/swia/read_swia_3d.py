# %% STATIC d1 mode reader
import os
import numpy as np
from py_space_zc import get_cdf_var

def read_swia_3d(filename_swia_3d):
    """
    Read and process MAVEN SWIA 3d data from a CDF file.

    Parameters
    ----------
    filename_swia_3d : str
        Full path to the SWIA 3d CDF file.

    Returns
    -------
    res : dict
        Dictionary containing processed SWIA 3d data with the following fields:
            - time       [num_time]                            : Time array in datetime64[ns]
            - energy     [num_time, num_energy]                : Energy table (sorted high -> low)
            - DEF        [num_time, num_energy, num_phi, num_theta] : differential energy flux
            - theta      [num_energy, num_theta]     : Elevation angles
            - phi        [num_phi]                             : Azimuthal angles [0, 360)

    """
    res = dict()
    if not os.path.isfile(filename_swia_3d):
        print(f"File not found: {filename_swia_3d}")
        return res

    try:
        # === Step 1: Read all variables in one call ===
        if "coarse" in filename_swia_3d:
            val_names = [
                'epoch', 'energy_coarse', 'diff_en_fluxes', 'theta_coarse', 'phi_coarse'
            ]
        elif "fine" in filename_swia_3d:
            val_names = [
                'epoch', 'energy_fine', 'diff_en_fluxes', 'theta_fine', 'phi_fine'
            ]

        is_time = [1] + [0] * (len(val_names) - 1)

        time, energy, DEF, theta, phi = get_cdf_var(
            cdf_filename = filename_swia_3d,
            variable_name=val_names,
            istime=is_time
        )

        # DEF:      [n_time, n_phi, n_theta, n_energy]

        theta = np.transpose(theta, [1,0])  # [ntheta, nenergy]
        DEF = np.transpose(DEF, [0, 3, 1, 2])         # [n_time, n_energy, n_phi, n_theta]

        # Store results (flipped energy axis)
        res = {"time": time,
               "energy": np.flip(energy),
               "DEF": np.flip(DEF, 1),
               "theta": np.flip(theta, 0),
               "phi": phi,}

        return res

    except Exception as e:
        print(f"Warning: Failed to read {filename_swia_3d}. Reason: {e}")

if __name__ == "__main__":
    path = 'F:\\data\\maven\\data\\sci\\swi\\l2\\2022\\06\\';
    filename_swia_3d = path + "mvn_swi_l2_coarsesvy3d_20220604_v02_r01.cdf"
    res = read_swia_3d(filename_swia_3d)
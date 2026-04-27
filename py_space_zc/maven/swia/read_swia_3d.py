# %% SWIA 3D mode reader
import os
import numpy as np
from py_space_zc import get_cdf_var


def average_pairs_1d(arr):
    """
    Average neighboring pairs in a 1D array.
    Example: (96,) -> (48,)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"average_pairs_1d expects 1D array, got shape {arr.shape}")
    if arr.size % 2 != 0:
        raise ValueError(f"Array length must be even, got {arr.size}")

    return np.nanmean(arr.reshape(-1, 2), axis=1)


def average_pairs_2d(arr):
    """
    Average 2x2 neighboring blocks in a 2D array.
    Example: (96,24) -> (48,12)
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"average_pairs_2d expects 2D array, got shape {arr.shape}")

    n0, n1 = arr.shape
    if n0 % 2 != 0 or n1 % 2 != 0:
        raise ValueError(f"Both dimensions must be even, got {arr.shape}")

    # (96,24) -> (48,2,12,2) -> mean over the paired axes
    return np.nanmean(
        np.nanmean(arr.reshape(n0 // 2, 2, n1 // 2, 2), axis=3),
        axis=1
    )


def read_swia_3d(filename_swia_3d):
    """
    Read and process MAVEN SWIA 3D data from a CDF file.

    Parameters
    ----------
    filename_swia_3d : str
        Full path to the SWIA 3D CDF file.

    Returns
    -------
    res : dict
        Dictionary containing processed SWIA 3D data with fields:
            - time   : [n_time]
            - energy : [n_energy]
            - DEF    : [n_time, n_energy, n_phi, n_theta]
            - theta  : [n_energy, n_theta]
            - phi    : [n_phi]
    """
    res = {}

    if not os.path.isfile(filename_swia_3d):
        print(f"File not found: {filename_swia_3d}")
        return res

    try:
        filename_lower = filename_swia_3d.lower()

        if "coarse" in filename_lower:
            val_names = [
                "epoch",
                "energy_coarse",
                "diff_en_fluxes",
                "theta_coarse",
                "phi_coarse",
            ]
            mode = "coarse"

        elif "fine" in filename_lower:
            val_names = [
                "epoch",
                "energy_fine",
                "diff_en_fluxes",
                "theta_fine",
                "phi_fine",
            ]
            mode = "fine"

        else:
            raise ValueError("Cannot determine SWIA mode from filename. Must contain 'fine' or 'coarse'.")

        is_time = [1] + [0] * (len(val_names) - 1)

        time, energy, DEF, theta, phi = get_cdf_var(
            cdf_filename=filename_swia_3d,
            variable_name=val_names,
            istime=is_time
        )

        energy = np.asarray(energy)
        DEF = np.asarray(DEF)
        theta = np.asarray(theta)
        phi = np.asarray(phi)

        # original DEF: [n_time, n_phi, n_theta, n_energy]
        # original theta: [ n_theta, n_energy]
        DEF = np.transpose(DEF, (0, 3, 1, 2)) #   [n_time, n_energy, n_phi, n_theta]
        theta = np.transpose(theta, (1, 0))   #   [n_energy, n_theta]

        if mode == "fine":
            # fine:
            # energy: (96,)      -> (48,)
            # theta : (96,24)    -> (48,12)
            # DEF   : already [nt,48,10,12]
            energy_table = average_pairs_1d(energy)
            theta_table = average_pairs_2d(theta)

            res = {
                "time": time,
                "energy": np.flip(energy_table, axis=0),
                "DEF": np.flip(DEF, axis=1),
                "theta": np.flip(theta_table, axis=0),
                "phi": phi,
            }

        elif mode == "coarse":
            res = {
                "time": time,
                "energy": np.flip(energy, axis=0),
                "DEF": np.flip(DEF, axis=1),
                "theta": np.flip(theta, axis=0),
                "phi": phi,
            }

        return res

    except Exception as e:
        print(f"Warning: Failed to read {filename_swia_3d}. Reason: {e}")
        return {}


if __name__ == "__main__":
    path = r"F:\data\maven\data\sci\swi\l2\2025\06"
    filename_swia_3d = os.path.join(path, "mvn_swi_l2_finesvy3d_20250615_v02_r01.cdf")
    res = read_swia_3d(filename_swia_3d)

    if res:
        print("Read successfully.")
        print("time shape  :", np.shape(res["time"]))
        print("energy shape:", np.shape(res["energy"]))
        print("DEF shape   :", np.shape(res["DEF"]))
        print("theta shape :", np.shape(res["theta"]))
        print("phi shape   :", np.shape(res["phi"]))
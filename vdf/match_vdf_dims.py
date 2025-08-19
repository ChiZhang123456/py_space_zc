import numpy as np

def match_vdf_dims(def_data, energy, phi, theta):
    """
    Normalize dimensions of energy, phi, and theta to match vdf dimensions.

    Author: Chi Zhang

    Parameters
    ----------
    def_data : np.ndarray
        4D array of shape (num_time, num_energies, num_phi, num_theta)
    energy : np.ndarray
        Energy levels array (1D or 2D)
    phi : np.ndarray
        Phi angles array (1D or 2D)
    theta : np.ndarray
        Theta angles array (1D, 2D or 3D)

    Returns
    -------
    energy_new : np.ndarray
        Energy levels with shape (num_time, num_energies)
    dE_new : np.ndarray
        Delta energy, shape (num_time, num_energies)
    phi_new : np.ndarray
        Phi angles, shape (num_time, num_phi)
    theta_new : np.ndarray
        Theta angles, shape (num_time, num_energies, num_theta)
    """

    num_time, num_energies, num_phi, num_theta = def_data.shape

    # --- Energy normalization ---
    energy = np.array(energy)
    if energy.ndim == 1 and energy.size == num_energies:
        energy_new = np.tile(energy[None, :], (num_time, 1))
    elif energy.shape == (num_time, num_energies):
        energy_new = energy
    else:
        raise ValueError(f"Incorrect dimensions for energy input: {energy.shape}")

    # Compute delta energy (keV or eV)
    dE_new = np.diff(energy_new, axis=1)
    dE_new = np.concatenate([dE_new, dE_new[:, -1][:, None]], axis=1)

    # --- Phi normalization ---
    phi = np.array(phi)
    if phi.ndim == 1 and phi.size == num_phi:
        phi_new = np.tile(phi[None, :], (num_time, 1))
    elif phi.shape == (num_time, num_phi):
        phi_new = phi
    else:
        raise ValueError(f"Incorrect dimensions for phi input: {phi.shape}")

    # --- Theta normalization ---
    theta = np.array(theta)
    if theta.ndim == 1 and theta.size == num_theta:
        theta_temp = np.tile(theta[None, :], (num_time * num_energies, 1))
        theta_new = theta_temp.reshape((num_time, num_energies, num_theta))
    elif theta.shape == (num_time, num_theta):
        theta_new = np.tile(theta[:, None, :], (1, num_energies, 1))
    elif theta.shape == (num_energies, num_theta):
        theta_new = np.tile(theta[None, :, :], (num_time, 1, 1))
    elif theta.shape == (num_time, num_energies, num_theta):
        theta_new = theta
    else:
        raise ValueError(f"Incorrect dimensions for theta input: {theta.shape}")

    return energy_new, dE_new, phi_new, theta_new

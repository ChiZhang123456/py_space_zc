# Import required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from py_space_zc import loadmat  # Custom loader for .mat files


def spu_yield(energy, ion='H'):
    """
    Interpolates the sputtering yield for a given ion species based on incident ion energy.

    Parameters:
    -----------
    energy : float or array-like
        The kinetic energy of the incident ions in electron volts (eV).

    ion : str, optional (default='H')
        The type of incident ion. Supported values (case-insensitive):
        - 'H', 'H+', 'h', 'h+': Proton
        - 'He', 'He++', 'he', 'he++': Alpha particle (Helium ion)
        - 'O+', 'o+': Singly-charged oxygen ion

    Returns:
    --------
    res : float or ndarray
        Interpolated sputtering yield corresponding to the input energy values.

    Notes:
    ------
    The interpolation is based on pre-calculated yield data stored in a MATLAB `.mat` file
    named 'ion_sputtering_energy_yield.mat'. The file must contain:
        - 'proton_energy', 'proton_yield'
        - 'alpha_energy', 'alpha_yield'
        - 'oxygen_energy', 'oxygen_yield'
    """

    # Load the yield data
    filename = 'ion_sputtering_energy_yield.mat'
    data = loadmat(filename)

    # Normalize ion name to lowercase for robust matching
    ion = ion.lower()

    # Match ion type to data arrays
    if ion in ['h', 'h+']:
        E = data['proton_energy']
        Y = data['proton_yield']
    elif ion in ['he', 'he++']:
        E = data['alpha_energy']
        Y = data['alpha_yield']
    elif ion in ['o+', 'o']:
        E = data['oxygen_energy']
        Y = data['oxygen_yield']
    else:
        raise ValueError(f"Unsupported ion type: {ion}. Supported types are: H, He, O+")

    # Perform interpolation
    interpolator = interp1d(E.reshape(-1), Y.reshape(-1),
                            kind='linear',bounds_error=False,)

    return interpolator(energy)

if __name__ == '__main__':
    energy = np.linspace(1.0, 10000.0, 10000)
    cs_h = spu_yield(energy, ion = 'H+')
    cs_o = spu_yield(energy, ion='O+')
    cs_he = spu_yield(energy, ion = 'He++')

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(energy, cs_h, label='H+', linewidth=2)
    ax.plot(energy, cs_he, label='He++', linewidth=2)
    ax.plot(energy, cs_o, label='O+', linewidth=2)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Yield', fontsize=12)
    ax.set_title('Sputtering Yield', fontsize=13)
    plt.grid(True, which='both', alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend(frameon=False)
    plt.show()


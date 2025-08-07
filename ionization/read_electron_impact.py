import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

def read_txt_data(filename):
    """
    Read a two-column .txt file containing tabulated data (e.g., electron impact cross sections).

    The file is expected to have two comma-separated columns:
    - Column 1: x-values (e.g., energy in eV)
    - Column 2: y-values (e.g., cross section in cm^2 or arbitrary units)

    Parameters
    ----------
    filename : str
        Full or relative path to the .txt file to read.

    Returns
    -------
    x : np.ndarray or None
        Array of x-values (first column). None if read fails.
    y : np.ndarray or None
        Array of y-values (second column). None if read fails.
    """
    try:
        data = np.loadtxt(filename, delimiter=',')
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("The file must contain exactly two columns.")
        x, y = data[:, 0], data[:, 1]
        return x, y
    except Exception as e:
        print(f"[ERROR] Failed to read '{filename}': {e}")
        return None, None


def EI_cross_section(energy, species):
    """
    Interpolate the electron impact cross section for a given species at specified energies.

    The function looks for a local .txt file containing tabulated cross section data
    for the target species. It performs linear interpolation over the energy range.

    Parameters
    ----------
    energy : float or np.ndarray
        Energy value(s) (in eV) at which to evaluate the interpolated cross section.
    species : str
        Atomic or molecular species (e.g., 'H', 'O2', 'CO2', case-insensitive).
        Both neutrals and ions are supported (e.g., 'O' and 'O+').

    Returns
    -------
    interpolated_values : float or np.ndarray or None
        Interpolated cross-section value(s) at the input energy.
        Returns None if the species is not supported or file reading fails.

    Notes
    -----
    - Required .txt files must be located in the current working directory.
    - File naming convention: e.g., 'CO2_EI.txt', 'O_EI.txt', etc.
    - Each file should contain two comma-separated columns: energy, cross section.
    """
    # Map species name to corresponding file
    species_map = {
        'h': 'H_EI.txt',
        'h+': 'H_EI.txt',
        'c': 'C_EI.txt',
        'c+': 'C_EI.txt',
        'o': 'O_EI.txt',
        'o+': 'O_EI.txt',
        'o2': 'O2_EI.txt',
        'o2+': 'O2_EI.txt',
        'co': 'CO_EI.txt',
        'co+': 'CO_EI.txt',
        'n2': 'N2_EI.txt',
        'n2+': 'N2_EI.txt',
        'co2': 'CO2_EI.txt',
        'co2+': 'CO2_EI.txt',
    }

    species_key = species.lower().strip()
    filename_only = species_map.get(species_key)

    if filename_only is None:
        print(f"[ERROR] No cross-section data for species: '{species}'")
        return None

    # Construct full path to the file (assuming it's in the current working directory)
    path = os.getcwd()
    filename = os.path.join(path, filename_only)

    # Read tabulated energy vs. cross-section data
    x_data, y_data = read_txt_data(filename)
    if x_data is None or y_data is None:
        return None

    # Create interpolation function and evaluate at the given energy
    try:
        interpolator = interp1d(
            x_data, y_data,
            kind='linear',
            bounds_error=False,)
        return interpolator(energy)
    except Exception as e:
        print(f"[ERROR] Interpolation failed: {e}")
        return None


# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    """
    Demonstration: load and visualize a cross-section file using logarithmic axes.
    """
    energy = np.logspace(1, 4, 1000)
    co_ei = EI_cross_section(energy, 'co')
    n2_ei = EI_cross_section(energy, 'n2')
    h_ei = EI_cross_section(energy, 'h')
    o_ei = EI_cross_section(energy, 'o')
    fig, ax = plt.subplots()

    # Plot the raw data
    ax.plot(energy, h_ei, color = 'red', label='H Cross Section')
    ax.plot(energy, o_ei, color = 'green', label='O Cross Section')
    ax.plot(energy, co_ei, color = 'blue', label='CO Cross Section')
    ax.plot(energy, n2_ei, color = 'violet', label='N2 Cross Section')

    # Set logarithmic scale for better visualization of wide-range data
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labeling
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Cross Section (cm² or a.u.)')
    ax.set_title('Electron Impact Cross Section')
    ax.legend()

    # Grid and layout
    ax.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()

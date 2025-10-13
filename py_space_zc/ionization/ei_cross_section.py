import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

import numpy as np

def read_txt_data(filename, from_x1e16=True):
    """
    读取 cross section 文本文件，兼容两种列格式：
      1) energy, elastic, ionization, others
      2) energy, ionization

    参数
    ----
    filename : str
        文件路径
    from_x1e16 : bool, default True
        如果数据列是以 x1e-16 归一（你的保存方式），则自动乘回 1e-16 得到 cm^2

    返回
    ----
    data : dict
        {
          'energy': np.ndarray (eV),
          'elastic': np.ndarray or None (cm^2),
          'ionization': np.ndarray or None (cm^2),
          'others': np.ndarray or None (cm^2)
        }
    """
    # 读取前几行，判断是否有表头、分隔符
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip() != '']

    if not lines:
        raise ValueError(f"Empty file: {filename}")

    # 判断首行是不是表头（含字母或括号等）
    first = lines[0]
    has_header = any(c.isalpha() for c in first)

    # 找一行数据行用于判断分隔符
    data_line = None
    for ln in lines:
        if any(ch.isdigit() for ch in ln):
            data_line = ln
            break
    if data_line is None:
        raise ValueError(f"No numeric data lines found in: {filename}")

    # 判断分隔符：若包含逗号，则用逗号，否则用任意空白
    delimiter = ',' if (',' in data_line) else None

    # 跳过表头行
    skiprows = 1 if has_header else 0
    arr = np.loadtxt(filename, delimiter=delimiter, ndmin=2, skiprows=skiprows)


    if arr.ndim != 2:
        arr = np.atleast_2d(arr)

    ncols = arr.shape[1]
    if ncols == 2:
        energy = arr[:, 0]
        elastic = None
        ionization = arr[:, 1]
        others = None
    elif ncols >= 4:
        energy = arr[:, 0]
        elastic = arr[:, 1]
        ionization = arr[:, 2]
        others = arr[:, 3]
    else:
        raise ValueError(f"Unsupported column count ({ncols}) in file: {filename}")

    # 若文件数值是以 x1e-16 保存，则乘回 1e-16 得到 cm^2
    scale = 1e-16 if from_x1e16 else 1.0
    def _scale(x):
        return None if x is None else x * scale

    return {
        'energy': energy,                 # eV
        'elastic': _scale(elastic),       # cm^2 或 None
        'ionization': _scale(ionization), # cm^2 或 None
        'others': _scale(others),         # cm^2 或 None
    }



def ei_cross_section(energy, species):
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
        'h': 'electron_impact_H.txt',
        'h+': 'electron_impact_H.txt',
        'c': 'electron_impact_C.txt',
        'c+': 'electron_impact_C.txt',
        'o': 'electron_impact_O.txt',
        'o+': 'electron_impact_O.txt',
        'o2': 'electron_impact_O2.txt',
        'o2+': 'electron_impact_O2.txt',
        'co': 'electron_impact_CO.txt',
        'co+': 'electron_impact_CO.txt',
        'n2': 'electron_impact_N2.txt',
        'n2+': 'electron_impact_N2.txt',
        'co2': 'electron_impact_CO2.txt',
        'co2+': 'electron_impact_CO2.txt',
        'ar': 'electron_impact_Ar.txt',
        'ar+': 'electron_impact_Ar.txt',
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
    res = read_txt_data(filename)
    E = res['energy']
    ionization = res['ionization']  # cm^2


    # Create interpolation function and evaluate at the given energy
    try:
        interpolator = interp1d(
            E, ionization,
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
    energy = np.logspace(0, 4, 10000)
    co_ei = ei_cross_section(energy, 'co')
    n2_ei = ei_cross_section(energy, 'n2')
    h_ei = ei_cross_section(energy, 'h')
    o_ei = ei_cross_section(energy, 'o')
    o2_ei = ei_cross_section(energy, 'o2')
    co2_ei = ei_cross_section(energy, 'co2')
    fig, ax = plt.subplots()

    # Plot the raw data
    ax.plot(energy, h_ei, color='red', label='H')
    ax.plot(energy, o_ei, color='forestgreen', label='O')
    ax.plot(energy, o2_ei, color='limegreen', label='O₂')
    ax.plot(energy, co_ei, color='navy', label='CO')
    ax.plot(energy, co2_ei, color='deepskyblue', label='CO₂')
    ax.plot(energy, n2_ei, color='purple', label='N₂')

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

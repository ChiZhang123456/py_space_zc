from .get_data import get_data
import numpy as np
import py_space_zc


def extract_data_minpa(tint, species):
    """
    Extracts and processes DEF (Differential Energy Flux) data for a given ion species
    from the 'minpa_mod1' dataset, and constructs a phase space distribution skymap.

    Parameters
    ----------
    tint : list or tuple of str
        Time interval of interest, typically in the form ['YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DDTHH:MM:SS'].

    species : str
        Ion species to extract. Supported values (case-insensitive):
            'H', 'H+', 'O', 'O+', 'O2', 'O2+', 'CO2', 'CO2+'

    Returns
    -------
    res : xarray.Dataset
        Phase space distribution skymap generated using `py_space_zc.vdf.create_pdist_skymap`.
        The dataset includes metadata and coordinates such as time, energy, angles, and DEF.

    Notes
    -----
    The DEF array has dimensions: [time, energy, phi, theta, mass_channel]
    Each species corresponds to a specific mass index in the 5th dimension.
    """

    # Mapping from species string (lowercase) to mass channel index
    species_mass_index = {
        'h': 1,
        'h+': 1,
        'o': 4,
        'o+': 4,
        'o2': 6,
        'o2+': 6,
        'co2': 7,
        'co2+': 7,
    }

    # Convert species to lowercase to handle case-insensitive input
    sp = species.lower()

    if sp not in species_mass_index:
        raise ValueError(f"Unsupported species '{species}'. Supported species are: {list(species_mass_index.keys())}")

    # Load data from source (using your custom get_data function)
    minpa = get_data(tint, 'minpa_mod1')

    # Extract DEF for the selected species
    # DEF shape: [time, energy, phi, theta, mass]
    mass_index = species_mass_index[sp]
    DEF = np.squeeze(minpa["DEF"][:, :, :, :, mass_index])

    # Create phase space distribution skymap using py_space_zc
    res = py_space_zc.vdf.create_pdist_skymap(
        time=minpa["time"],  # 1D time array
        energy=minpa["energy"],  # 1D energy array
        data=DEF,  # 4D DEF array: [time, energy, phi, theta]
        phi=minpa["phi"],  # 1D azimuthal angle array
        theta=minpa["theta"],  # 1D polar angle array
        Units = "keV/(cm^2 s sr keV)",  # Units for DEF
        species = species.upper(),  # Nominal species label (may be updated later)
        direction_is_velocity=True  # Indicates velocity vector direction
    )

    return res

from scipy import constants


def get_particle_mass_charge(species):
    """
    Get particle mass and charge based on species type.

    Parameters:
    -----------
    species : str
        String specifying the particle species. Supported types include:
        - Hydrogen: 'H', 'H+', 'ion', 'ions', 'p', 'proton'
        - Helium: 'He++', 'He+', 'alpha', 'helium'
        - Oxygen: 'O', 'O+', 'oxygen'
        - Oxygen molecule: 'O2', 'O2+', 'oxygen2'
        - Carbon dioxide: 'CO2', 'CO2+', 'carbondioxide'
        - Electron: 'e', 'electron', 'e-'

    Returns:
    --------
    tuple : (mass, charge)
        mass : float
            Particle mass in kg
        charge : float
            Particle charge in Coulombs

    Raises:
    -------
    ValueError
        If the species is not recognized

    Examples:
    ---------
    >>> mass, charge = get_particle_properties('H+')
    >>> mass, charge = get_particle_properties('he++')
    >>> mass, charge = get_particle_properties('electron')
    """

    species_normalized = species.lower().strip()

    # Hydrogen/Proton
    if species_normalized in ['h', 'h+', 'ion', 'ions', 'p', 'proton']:
        mass = constants.proton_mass
        charge = constants.elementary_charge

    # Helium (He++ alpha particles or He+)
    elif species_normalized in ['he++', 'he+', 'alpha', 'helium']:
        mass = 4 * constants.proton_mass
        if species_normalized == 'he+':
            charge = constants.elementary_charge  # He+ (singly ionized)
        else:
            charge = 2 * constants.elementary_charge  # He++ (alpha particle)

    # Oxygen
    elif species_normalized in ['o', 'o+', 'oxygen']:
        mass = 16 * constants.proton_mass
        charge = constants.elementary_charge

    # Oxygen molecule
    elif species_normalized in ['o2', 'o2+', 'oxygen2']:
        mass = 32 * constants.proton_mass
        charge = constants.elementary_charge

    # Carbon dioxide
    elif species_normalized in ['co2', 'co2+', 'carbondioxide']:
        mass = 44 * constants.proton_mass
        charge = constants.elementary_charge

    # Electron
    elif species_normalized in ['e', 'electron', 'e-']:
        mass = constants.electron_mass
        charge = -constants.elementary_charge  # Negative charge for electrons

    else:
        raise ValueError(f"Unknown species: {species}. Supported species are: "
                         "H/H+/proton, He++/He+/alpha, O/O+, O2/O2+, CO2/CO2+, e/electron")

    return mass, charge


if __name__ == "__main__":
    mass, charge = get_particle_mass_charge("O+")
import numpy as np
from .get_particle_mass_charge import get_particle_mass_charge

def convert_energy_velocity(data, direction: str, species: str = 'H+') -> np.ndarray:
    """
    Convert between kinetic energy (eV) and velocity (km/s) for a given ion species.

    Author: Chi Zhang

    Parameters
    ----------
    data : array-like
        Energy in eV (for 'E>V') or velocity in km/s (for 'V>E').
    direction : str
        Conversion direction:
            'E>V', 'e>v', 'E2V', or 'e2v' for energy -> velocity
            'V>E', 'v>e', 'V2E', or 'v2e' for velocity -> energy
    species : str, optional
        Ion species, default is 'H+' (proton).
        Supported: 'H', 'H+', 'He++', 'O', 'O2', 'e'

    Returns
    -------
    res : np.ndarray
        Converted values: velocity in km/s or energy in eV.
    """
    species = species.lower()
    m, q = get_particle_mass_charge(species)
    data = np.asarray(data)
    if direction.lower() in ['e>v', 'e2v']:
        res = np.sqrt(2 * q * data / m) * 1e-3  # [km/s]
    elif direction.lower() in ['v>e', 'v2e']:
        res = (data * 1e3) ** 2 * m / (2 * q)   # [eV]
    else:
        raise ValueError(f"Invalid conversion direction: {direction}")

    return res


if __name__ == "__main__":
    energy = 1000
    velocity = convert_energy_velocity(energy, direction='e2v', species='H+')
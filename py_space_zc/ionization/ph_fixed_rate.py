import numpy as np

def CO2_hv(nCO2_m3):
    """
    Photochemical ionization of CO₂: CO₂ + hν → CO₂⁺ + e⁻

    Parameters
    ----------
    nCO2_m3 : float or np.ndarray
        CO₂ neutral density in [m⁻³]

    Returns
    -------
    ion_rate : float or np.ndarray
        Ion production rate in [m⁻³/s]
    """
    return 1.83e-6 * nCO2_m3


def CO2_hv_diss(nCO2_m3):
    """
    Photodissociative ionization of CO₂: CO₂ + hν → O⁺ + CO⁺ + e⁻

    Parameters
    ----------
    nCO2_m3 : float or np.ndarray
        CO₂ neutral density in [m⁻³]

    Returns
    -------
    ion_rate : float or np.ndarray
        Ion production rate in [m⁻³/s]
    """
    return 1.65e-7 * nCO2_m3


def O_hv(nO_m3):
    """
    Photochemical ionization of O: O + hν → O⁺ + e⁻

    Parameters
    ----------
    nO_m3 : float or np.ndarray
        Atomic oxygen neutral density in [m⁻³]

    Returns
    -------
    ion_rate : float or np.ndarray
        Ion production rate in [m⁻³/s]
    """
    return 3.41e-7 * nO_m3

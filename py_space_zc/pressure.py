import numpy as np

def dynamic_pressure(n, Vn):
    """
    Calculate the dynamic pressure of a plasma.

    Parameters
    ----------
    n : float or ndarray
        Plasma number density in cm⁻³.
    Vn : float or ndarray
        Bulk velocity in km/s.

    Returns
    -------
    p_dyn : float or ndarray
        Dynamic pressure in nanoPascal (nPa).

    Notes
    -----
    Uses the formula:
        P_dyn [nPa] = 1.6726 × 10⁻⁶ × n [cm⁻³] × Vn² [km²/s²]
    where 1.6726 × 10⁻²⁷ kg is the mass of a proton,
    and appropriate unit conversions yield pressure in nPa.
    """
    return 1.6726e-6 * n * Vn**2  # nPa


def thermal_pressure(n, T):
    """
    Calculate the thermal pressure of a plasma.

    Parameters
    ----------
    n : float or ndarray
        Number density in cm⁻³.
    T : float or ndarray
        Temperature in electron-volts (eV).

    Returns
    -------
    p_th : float or ndarray
        Thermal pressure in nanoPascal (nPa).

    Notes
    -----
    Uses the formula:
        P_th [nPa] = n [cm⁻³] × 1e6 × T [eV] × 1.6e-19 [J/eV] × 1e9
                   = n × T × 1.6e-4
    where:
        - n is converted to m⁻³ (×1e6)
        - T is converted to Joules (×1.6e-19)
        - result is converted to nPa (×1e9)
    """
    return n * T * 1.6e-4  # nPa

def magnetic_pressure(Bt):
    """
    Calculate magnetic pressure.

    Parameters
    ----------
    Bt : float or ndarray
        Magnetic field magnitude in nT.

    Returns
    -------
    p_mag : float or ndarray
        Magnetic pressure in nanoPascal (nPa).

    Notes
    -----
    Formula:
        P_mag [nPa] = (Bt / 50)²
    """
    return (Bt / 50.0) ** 2  # [nPa]

def pressure(n=None, Vn=None, T=None, Bt=None, option='pdy'):
    """
    Compute plasma pressure: dynamic, thermal, or magnetic.

    Parameters
    ----------
    n : float or ndarray, optional
        Number density in cm⁻³.
    Vn : float or ndarray, optional
        Velocity in km/s. Required for 'pdy'.
    T : float or ndarray, optional
        Temperature in eV. Required for 'pth'.
    Bt : float or ndarray, optional
        Magnetic field magnitude in nT. Required for 'pmag'.
    option : str, optional
        Pressure type: one of {'pdy', 'pth', 'pmag'}.

    Returns
    -------
    pressure : float or ndarray
        Pressure in nanoPascal (nPa).

    Raises
    ------
    ValueError
        If required inputs are missing or option is invalid.
    """
    option = option.lower()

    if option == 'pdy':
        if n is None or Vn is None:
            raise ValueError("Both n and Vn are required for dynamic pressure ('pdy').")
        return dynamic_pressure(n, Vn)

    elif option == 'pth':
        if n is None or T is None:
            raise ValueError("Both n and T are required for thermal pressure ('pth').")
        return thermal_pressure(n, T)

    elif option == 'pmag':
        if Bt is None:
            raise ValueError("Bt is required for magnetic pressure ('pmag').")
        return magnetic_pressure(Bt)

    else:
        raise ValueError(f"Unknown option '{option}'. Supported options: 'pdy', 'pth', 'pmag'.")


if __name__ == '__main__':
    n = 3.0  # cm^-3
    T = 100.0  # eV
    Vn = 400.0  # km/s
    Bt = 10.0  # nT
    print(f"Dynamic pressure:  {pressure(n = n, Vn = Vn, option='pdy'):.2f} nPa")
    print(f"Thermal pressure:  {pressure(n = n, T = T, option='pth'):.2f} nPa")
    print(f"Magnetic pressure: {pressure(Bt = Bt, option='pmag'):.2f} nPa")

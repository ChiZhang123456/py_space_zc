import numpy as np

def hybrid_alfven_speed(B1_nT, B2_nT, n1_cm3, n2_cm3, ion_mass_kg=1.67262192369e-27):
    """
    Calculate hybrid (asymmetric) Alfven speed in km/s.

    Parameters
    ----------
    B1_nT : float or array
        Magnetic field on side 1 (nT)
    B2_nT : float or array
        Magnetic field on side 2 (nT)
    n1_cm3 : float or array
        Number density on side 1 (cm^-3)
    n2_cm3 : float or array
        Number density on side 2 (cm^-3)
    ion_mass_kg : float
        Ion mass (default: proton mass)

    Returns
    -------
    Va_km_s : float or array
        Hybrid Alfven speed (km/s)
    """

    mu0 = 4 * np.pi * 1e-7  # vacuum permeability

    # Unit conversions
    B1 = B1_nT * 1e-9       # nT -> T
    B2 = B2_nT * 1e-9
    n1 = n1_cm3 * 1e6       # cm^-3 -> m^-3
    n2 = n2_cm3 * 1e6

    # Mass densities
    rho1 = n1 * ion_mass_kg
    rho2 = n2 * ion_mass_kg

    # Hybrid Alfven speed (Cassak & Shay 2007)
    Va = np.sqrt(B1 * B2 * (B1 + B2) / (mu0 * (rho1 * B2 + rho2 * B1)))

    return Va / 1e3  # convert to km/s

if __name__ == '__main__':
    Va = hybrid_alfven_speed(3.93   , 3.8, 5.4, 5.7)
    print(f"{Va:.2f} km/s")
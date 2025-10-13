import numpy as np

def gyro_information(B, T, species='e'):
    """
    Compute thermal speed [km/s], gyrofrequency [Hz], and gyroradius [km]
    for a given particle species in a magnetic field.

    Parameters
    ----------
    B : float or ndarray
        Magnetic field strength in nT.
    T : float or ndarray
        Characteristic energy (temperature) in eV.
    species : str
        Particle type: one of ['e', 'H', 'H+', 'O', 'O+', 'O2+', 'CO2', 'CO2+'] (case-insensitive).

    Returns
    -------
    v_th : float or ndarray
        Thermal speed in km/s.
    f_gyro : float or ndarray
        Gyrofrequency in Hz.
    gyroradius : float or ndarray
        Gyroradius in kilometers.

    Raises
    ------
    ValueError
        If the species is not recognized.
    """

    B = np.asarray(B)  # [nT]
    T = np.asarray(T)  # [eV]
    sp = species.strip().upper()

    if sp == 'E':
        v_th = 593 * np.sqrt(T)                # [km/s]
        f_gyro = 28 * B                        # [Hz]
        gyroradius = 3.4 * np.sqrt(T) / B      # [km]

    elif sp in ['H', 'H+']:
        v_th = 13.8 * np.sqrt(T)               # [km/s]
        f_gyro = 0.0153 * B                    # [Hz]
        gyroradius = 144 * np.sqrt(T) / B      # [km]

    elif sp in ['O', 'O+']:
        v_th = 3.46 * np.sqrt(T)               # [km/s]
        f_gyro_mHz = 0.95 * B                  # [mHz]
        f_gyro = f_gyro_mHz * 1e-3             # [Hz]
        gyroradius = 578 * np.sqrt(T) / B      # [km]

    elif sp == 'O2+':
        v_th = 2.44 * np.sqrt(T)               # [km/s]
        f_gyro_mHz = 0.475 * B                 # [mHz]
        f_gyro = f_gyro_mHz * 1e-3             # [Hz]
        gyroradius = 817.4 * np.sqrt(T) / B    # [km]

    elif sp == 'CO2+':
        v_th = 2.08 * np.sqrt(T)               # [km/s]
        f_gyro_mHz = 3.477 * B                 # [mHz]
        f_gyro = f_gyro_mHz * 1e-3             # [Hz]
        gyroradius = 955.2 * np.sqrt(T) / B    # [km]

    else:
        raise ValueError(f"Unknown species: '{species}'. Supported: 'e', 'H+', 'O+', 'O2+'.")

    return v_th, f_gyro, gyroradius


if __name__ == '__main__':
    B = 10.0  # nT
    T = 200.0  # eV

    for s in ['e', 'H+', 'O+', 'O2+']:
        v, f, r = gyro_information(B, T, s)
        print(f"{s.upper():>4} → Speed = {v:.1f} km/s | f_gyro = {f:.4f} Hz | r_gyro = {r:.1f} km")

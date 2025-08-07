import numpy as np
import py_space_zc

def extract_data_c6(tint, species):
    """
    Get MAVEN STATIC C6 differential energy flux (DEF) spectrogram
    for a selected ion species within a given time interval.

    Parameters
    ----------
    tint : list of str
        Time interval [start_time, end_time], e.g., ["2017-02-07T10:01:00", "2017-02-07T11:00:00"]
    species : str
        Ion species to extract, e.g., "H+", "He+", "O+", "O2+", "CO2+"

    Returns
    -------
    data_ts : xarray.DataArray
        2D spectrogram with DEF [keV/(cm^2 s sr keV)] in dimensions (time, energy)
    """

    # Load STATIC C6 data
    c6 = py_space_zc.maven.get_data(tint, "static_c6")

    # Normalize species string
    sp = species.lower()

    # Define mass range and LaTeX label
    if sp in ["h", "h+", "p"]:
        mass_range = [0.0, 1.55]
    elif sp in ["he", "he+", "he++"]:
        mass_range = [1.55, 2.7]
    elif sp in ["o", "o+"]:
        mass_range = [14.0, 20.0]
    elif sp in ["o2", "o2+"]:
        mass_range = [24.0, 40.0]
    elif sp in ["co2", "co2+"]:
        mass_range = [40.0, 60.0]
    else:
        raise ValueError(f"Unsupported species: {species}")

    # Select mass channels
    mass = np.array(c6["mass"])  # shape: (nmass,)
    mask = (mass >= mass_range[0]) & (mass <= mass_range[1])

    # Sum DEF over selected mass bins
    def_omni = np.nansum(c6["DEF"][:, :, mask], axis=2)

    # Construct xarray DataArray using pyrf.ts_spectr
    data_ts = py_space_zc.ts_spectr(
        time=c6["time"],
        ener=c6["energy"],
        data=def_omni,
        comp_name="energy",
        attrs={
            "species": species,
            "UNITS": "keV/(cm^2 s sr keV)",
            "long_name": f"STATIC C6 DEF for {species}"
        },
    )

    return data_ts
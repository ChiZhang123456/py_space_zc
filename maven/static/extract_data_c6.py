import numpy as np
from py_space_zc import maven, ts_spectr
import copy

def extract_data_c6(tint, species, correct_background=False):
    """
    Get MAVEN STATIC C6 differential energy flux (DEF) spectrogram
    for a selected ion species within a given time interval.

    Parameters
    ----------
    tint : list of str
        Time interval [start_time, end_time], e.g., ["2017-02-07T10:01:00", "2017-02-07T11:00:00"]
    species : str
        Ion species to extract, e.g., "H+", "He+", "O+", "O2+", "CO2+"
    correct_background : bool, optional
        If True, corrects for background in the STATIC C6 data. Default is False.

    Returns
    -------
    data_ts : xarray.DataArray
        2D spectrogram with DEF [keV/(cm^2 s sr keV)] in dimensions (time, energy)
    """

    # Load STATIC C6 data
    c6_raw = maven.get_data(tint, "static_c6")

    # Correct background if requested
    if correct_background:
        c6 = maven.static.correct_bkg_c6(c6_raw)
    else:
        c6 = copy.deepcopy(c6_raw)

    # Normalize species string
    sp = species.lower()

    # Define mass ranges based on species
    species_mass_ranges = {
        "h": [0.0, 1.55], "h+": [0.0, 1.55], "p": [0.0, 1.55],
        "he": [1.55, 2.7], "he+": [1.55, 2.7], "he++": [1.55, 2.7],
        "o": [14.0, 20.0], "o+": [14.0, 20.0],
        "o2": [24.0, 40.0], "o2+": [24.0, 40.0],
        "co2": [40.0, 60.0], "co2+": [40.0, 60.0]
    }

    # Handle unsupported species
    if sp not in species_mass_ranges:
        raise ValueError(
            f"Unsupported species: {species}. Supported species are: {', '.join(species_mass_ranges.keys())}")

    # Get mass range for the selected species
    mass_range = species_mass_ranges[sp]

    # Select mass channels and create a mask
    mass = np.array(c6["mass"])  # shape: (nmass,)
    mask = (mass >= mass_range[0]) & (mass <= mass_range[1])

    # Sum DEF over selected mass bins
    def_omni = np.nansum(c6["DEF"][:, :, mask], axis=2)

    # Construct xarray DataArray using pyrf.ts_spectr
    data_ts = ts_spectr(
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

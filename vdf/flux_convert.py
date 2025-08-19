from .def2psd import def2psd
from .psd2def import psd2def
from .dpf2psd import dpf2psd
from .psd2dpf import psd2dpf


def flux_convert(inp, option: str = "def2psd"):
    """
    Converts particle flux data between different physical units.

    Parameters
    ----------
    inp : xarray.DataArray or compatible object
        Input data to be converted. This is typically an energy-time spectrogram
        in one of the following forms:
            - DEF: Differential Energy Flux [keV/(cm^2 s sr keV)]
            - PSD: Phase Space Density [s^3/m^6]
            - DPF: Differential Particle Flux [1/(cm^2 s sr eV)]

    option : str, default = "def2psd"
        The type of conversion to perform. Supported values:
            - "def2psd" : Convert from DEF to PSD
            - "psd2def" : Convert from PSD to DEF
            - "psd2dpf" : Convert from PSD to DPF
            - "dpf2psd" : Convert from DPF to PSD

    Returns
    -------
    res : xarray.DataArray
        The converted data, in the corresponding output unit.

    Raises
    ------
    ValueError
        If the 'option' provided is not recognized.

    Notes
    -----
    This is a simple wrapper around `pyrfu.mms` conversion utilities.
    It is assumed that the input has appropriate metadata (e.g., energy levels)
    for the conversion to succeed.
    """

    # Select conversion based on option
    if option == "def2psd":
        res = def2psd(inp)
        
    elif option == "psd2def":
        res = psd2def(inp)
        
    elif option == "psd2dpf":
        res = psd2dpf(inp)
        
    elif option == "dpf2psd":
        res = dpf2psd(inp)
        
    else:
        raise ValueError(f"Unsupported conversion option '{option}'. "
                         "Choose from 'def2psd', 'psd2def', 'psd2dpf', or 'dpf2psd'.")

    return res

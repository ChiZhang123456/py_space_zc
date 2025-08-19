import numpy as np
from py_space_zc import maven, vdf, ts_vec_xyz
from pyrfu.pyrf import extend_tint, resample
import copy

def moments_d1(d1_raw, species, Emin = 0.0, Emax = 500000.0, 
               correct_background=False, correct_vsc=False, ):
    """
    Compute plasma moments from MAVEN STATIC D1 data for a given ion species.

    Parameters
    ----------
    d1 : dict-like
        STATIC D1 dataset containing:
        - 'time'   : time array
        - 'H_DEF', 'O_DEF', 'O2_DEF' : differential energy flux arrays
        - 'scpot'  : spacecraft potential time series
        - 'sta2mso': rotation matrices STATIC→MSO
        - possibly other keys needed for maven.static utilities

    species : str
        Ion species to process. Accepted values (case-insensitive):
        {"H", "H+", "O", "O+", "O2", "O2+"}

    correct_background : bool, default False
        If True, apply STATIC D1 background correction.

    correct_vsc : bool, default False
        If True, estimate spacecraft velocity in STATIC frame and apply correction.

    Returns
    -------
    dict
        Plasma moments as returned by vdf.moments_calculation
        (number density, bulk velocity, pressure tensor, temperature tensor, fluxes).
    """

    # 1. Optional STATIC D1 background correction
    if correct_background:
        d1 = maven.static.correct_bkg_d1(d1_raw)
    else:
        d1 = copy.deepcopy(d1_raw)


    # 2. Select species and convert DEF → PSD
    s = species.lower()
    if s in {"h", "h+"}:
        vdf_i = vdf.flux_convert(d1["H_DEF"], "def2psd")
        
    elif s in {"o", "o+"}:
        vdf_i = vdf.flux_convert(d1["O_DEF"], "def2psd")
        
    elif s in {"o2", "o2+"}:
        vdf_i = vdf.flux_convert(d1["O2_DEF"], "def2psd")
        
    else:
        raise ValueError(f"Unsupported species: {species}")

    # 3. Optional spacecraft velocity correction
    if correct_vsc:
        vsc_mso = maven.get_vsc_mso(d1["time"])
        vsc_mso = ts_vec_xyz(d1["time"], vsc_mso)

        # Transform spacecraft velocity from MSO to STATIC frame
        vsc_sta = maven.static.mso2sta(vsc_mso, d1)
    else:
        # Zero spacecraft velocity if no correction
        vsc_sta = ts_vec_xyz(d1["time"],np.zeros((len(d1["time"]), 3), dtype=np.float64))

    # 4. Compute plasma moments
    moments = vdf.moments_calculation(
        vdf_i,
        sc_pot=d1["scpot"],
        vsc_instrument = vsc_sta,
        Emin = Emin,
        Emax = Emax
    )

    # convert STATIC coordinates to MSO coordiantes
    moments["V"] = maven.static.sta2mso(moments["V"], d1)
    moments["H"] = maven.static.sta2mso(moments["H"], d1)
    moments["Q"] = maven.static.sta2mso(moments["Q"], d1)
    moments["K"] = maven.static.sta2mso(moments["K"], d1)

    return moments

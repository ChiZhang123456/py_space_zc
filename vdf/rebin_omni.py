import numpy as np
import logging
import pyrfu.pyrf as pyrf
from py_space_zc import ts_spectr, ts_scalar
from .convert_energy_velocity import convert_energy_velocity
from .create_pdist_skymap import create_pdist_skymap
from .expand_4d_grid import expand_4d_grid
from .vxyz_from_polar import vxyz_from_polar
from .flux_convert import flux_convert

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebin_omni(inp_omni, scpot=None):
    """
    Rebins omni-directional phase space density (PSD) onto a new energy grid.
    Applies spacecraft potential correction, interpolates PSD values into new
    energy bins, and returns data in original units.

    Parameters
    ----------
    inp_omni : ts_spectr
        Input omni-directional PSD or DEF in shape [time, energy].
        Units must be one of:
            - 'keV/(cm^2 s sr keV)' (DEF)
            - '1/(cm^2 s sr eV)'     (DPF)
            - 's^3/m^6'              (PSD)

    scpot : ts_scalar or None, optional
        Spacecraft potential [V]. If None, assumes 0V correction.

    Returns
    -------
    res : ts_spectr
        Rebinned omni data with the same unit type as input (DEF, DPF, or PSD).
    """

    # 1. Define new target energy grid (eV)
    energy_new = 10 ** np.linspace(np.log10(1), np.log10(25000), 40)
    dE = np.diff(energy_new, append=energy_new[-1])

    # 2. Extract input metadata
    time = inp_omni.time
    energy_old = inp_omni.energy.values
    unit = inp_omni.attrs.get('UNITS', '').lower()
    species = inp_omni.attrs.get('SPECIES', 'H+')

    # 3. Convert to PSD if necessary
    if unit in ['kev/(cm^2 s sr kev)', 'ev/(cm^2 s sr ev)', '1/(cm^2 s sr)']:
        omni_psd = flux_convert(inp_omni, 'def2psd')
    elif unit in ['1/(cm^2 s sr ev)', '1/(cm^2 s sr kev)']:
        omni_psd = flux_convert(inp_omni, 'dpf2psd')
    elif unit in ['s^3/m^6']:
        omni_psd = inp_omni
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    omni_psd_data = omni_psd.data
    num_time, num_energy = omni_psd_data.shape

    # 4. Apply spacecraft potential correction
    if scpot is None:
        scpot_val = np.zeros(len(time))  # No correction
    else:
        scpot_val = pyrf.resample(scpot, omni_psd.time).data

    # 5. Shift energy by spacecraft potential
    if energy_old.ndim == 2:
        energymat = energy_old - scpot_val[:, None]
    elif energy_old.ndim == 1:
        energymat = energy_old - scpot_val
    else:
        raise ValueError("Unexpected energy dimension")

    # Remove negative energy values
    energymat = np.where(energymat > 0, energymat, np.nan)

    # 6. Initialize output array
    omni_new = np.zeros((num_time, len(energy_new)))

    logger.info("Begin rebinning the omni-directional data...")

    # 7. Loop over each time step and rebin
    for it in range(num_time):
        temp_data = omni_psd_data[it]
        temp_energy = energymat[it]
        for iE, E in enumerate(energy_new):
            mask = (temp_energy >= E) & (temp_energy < E + dE[iE])
            omni_new[it, iE] = np.nansum(temp_data[mask])

    logger.info("Finished rebinning.")

    omni_rebin = ts_spectr(
        time = time.data,
        ener = energy_new,
        data = omni_new,
        attrs = {"UNITS": "s^3/m^6",
                 "species":species})

    # 8. Convert back to original units
    if unit in ['kev/(cm^2 s sr kev)', 'ev/(cm^2 s sr ev)', '1/(cm^2 s sr)']:
        res = flux_convert(omni_rebin, 'psd2def')
    elif unit in ['1/(cm^2 s sr ev)', '1/(cm^2 s sr kev)']:
        res = flux_convert(omni_rebin, 'psd2dpf')
    elif unit in ['s^3/m^6']:
        res = omni_rebin

    return res

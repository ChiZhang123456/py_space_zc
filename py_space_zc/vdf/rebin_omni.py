import numpy as np
import logging
import pyrfu.pyrf as pyrf
import numba
from numba import njit, prange
from py_space_zc import ts_spectr, ts_scalar
from .flux_convert import flux_convert

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@njit(parallel=True, cache=True)
def _rebin_omni_kernel(psd_data, energymat_corr, dE_old, e_edges, num_time, nE_new):
    """
    Numba kernel for omni-directional rebinning using volume-weighted averaging.

    Logic:
    f_new = Sum(f_old * weight_old) / Sum(weight_old)
    where weight = sqrt(E) * dE
    """
    res_accum = np.zeros((num_time, nE_new), dtype=np.float64)
    vol_accum = np.zeros((num_time, nE_new), dtype=np.float64)

    num_energy_old = psd_data.shape[1]

    for it in prange(num_time):
        for ie in range(num_energy_old):
            f_val = psd_data[it, ie]
            e_val = energymat_corr[it, ie]
            de_val = dE_old[it, ie]

            # Skip invalid or negative data
            if np.isnan(f_val) or f_val <= 0 or np.isnan(e_val) or e_val <= 0:
                continue

            # Volume weight for 1D omni data: sqrt(E) * dE
            weight = np.sqrt(e_val) * de_val

            # Boundary check for target grid
            if e_val < e_edges[0] or e_val >= e_edges[-1]:
                continue

            # Find target bin index
            idx = np.searchsorted(e_edges, e_val) - 1

            res_accum[it, idx] += f_val * weight
            vol_accum[it, idx] += weight

    # Final Normalization: Intensive property preservation
    psd_new = np.zeros((num_time, nE_new), dtype=np.float64)
    for it in prange(num_time):
        for ie in range(nE_new):
            if vol_accum[it, ie] > 0:
                psd_new[it, ie] = res_accum[it, ie] / vol_accum[it, ie]
            else:
                psd_new[it, ie] = np.nan

    return psd_new


def rebin_omni(inp_omni, scpot=None, energy_new=None):
    """
    Rebins omni-directional PSD using volume-weighted averaging to prevent
    density distortion when target bins are smaller than instrument bins.

    Parameters
    ----------
    inp_omni : ts_spectr
        Input omni-directional data [time, energy].
    scpot : ts_scalar or None, optional
        Spacecraft potential [V] for energy correction.
    energy_new : array_like, optional
        Target energy bin centers [eV]. Default: 1-25000 eV (40 bins).

    Returns
    -------
    res : ts_spectr
        Rebinned omni data in the original unit type.
    """

    # 1. Define target energy grid and edges
    if energy_new is None:
        energy_new = 10 ** np.linspace(np.log10(1), np.log10(25000), 40)

    # Calculate edges for the kernel binary search
    de_target = np.diff(energy_new)
    e_edges = np.concatenate(([energy_new[0] - de_target[0] / 2],
                              energy_new[:-1] + de_target / 2,
                              [energy_new[-1] + de_target[-1] / 2]))

    # 2. Extract input metadata
    time = inp_omni.time
    energy_old = inp_omni.energy.values
    unit = inp_omni.attrs.get('UNITS', '').lower()
    species = inp_omni.attrs.get('SPECIES', 'H+')

    # 3. Convert input to PSD (s^3/m^6) for physical weighting
    if unit in ['kev/(cm^2 s sr kev)', 'ev/(cm^2 s sr ev)', '1/(cm^2 s sr)']:
        omni_psd = flux_convert(inp_omni, 'def2psd')
    elif unit in ['1/(cm^2 s sr ev)', '1/(cm^2 s sr kev)']:
        omni_psd = flux_convert(inp_omni, 'dpf2psd')
    elif unit in ['s^3/m^6']:
        omni_psd = inp_omni
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    psd_data = omni_psd.data
    num_time = len(time)

    # 4. Calculate instrument energy widths (dE_old)
    if energy_old.ndim == 1:
        temp_dE = np.abs(np.gradient(energy_old))
        dE_old = np.tile(temp_dE, (num_time, 1))
        energy_old_mat = np.tile(energy_old, (num_time, 1))
    else:
        dE_old = np.abs(np.gradient(energy_old, axis=1))
        energy_old_mat = energy_old

    # 5. Spacecraft Potential Correction
    if scpot is None:
        scpot_val = np.zeros(num_time)
    else:
        scpot_val = pyrf.resample(scpot, omni_psd.time).data

    energymat_corr = energy_old_mat - scpot_val[:, None]

    # 6. Perform rebinning via Numba Kernel
    logger.info(f"Rebinning {num_time} steps of omni data via volume-weighted averaging...")

    psd_new_data = _rebin_kernel_numba(
        psd_data, energymat_corr, dE_old, e_edges, num_time, len(energy_new)
    )

    # 7. Construct output spectr
    omni_rebin = ts_spectr(
        time=time.data,
        ener=energy_new,
        data=psd_new_data,
        attrs={"UNITS": "s^3/m^6", "species": species}
    )

    # 8. Convert back to the original units
    logger.info("Finished rebinning. Converting back to original units...")
    if unit in ['kev/(cm^2 s sr kev)', 'ev/(cm^2 s sr ev)', '1/(cm^2 s sr)']:
        res = flux_convert(omni_rebin, 'psd2def')
    elif unit in ['1/(cm^2 s sr ev)', '1/(cm^2 s sr kev)']:
        res = flux_convert(omni_rebin, 'psd2dpf')
    else:
        res = omni_rebin

    return res
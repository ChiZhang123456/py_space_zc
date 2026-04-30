import numpy as np
import logging
import numba
from numba import njit, prange
import pyrfu.pyrf as pyrf

# Assuming these are your local utility functions
from .convert_energy_velocity import convert_energy_velocity
from .create_pdist_skymap import create_pdist_skymap
from .expand_4d_grid import expand_4d_grid
from .vxyz_from_polar import vxyz_from_polar

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@njit(parallel=True, cache=True)
def _rebin_kernel_numba(psd_weighted, d3v_old, Emat, az, el,
                        e_edges, az_edges, el_edges,
                        num_time, nE_new, nP_new, nT_new):
    """
    Numba-accelerated kernel for 3D rebinning using volume-weighted averaging.

    Logic:
    f_new = Sum(f_old * d3v_old) / Sum(d3v_old_captured)
    This prevents density spikes when target bins are smaller than instrument bins.
    """
    # Accumulator for weighted PSD (numerator)
    res_accum = np.zeros((num_time, nE_new, nP_new, nT_new), dtype=np.float64)
    # Accumulator for captured volume (denominator)
    vol_accum = np.zeros((num_time, nE_new, nP_new, nT_new), dtype=np.float64)

    _, nE_old, nP_old, nT_old = psd_weighted.shape

    for it in prange(num_time):
        for ie in range(nE_old):
            for ip in range(nP_old):
                for il in range(nT_old):

                    val = psd_weighted[it, ie, ip, il]
                    vol = d3v_old[it, ie, ip, il]

                    if np.isnan(val) or val <= 0 or np.isnan(vol):
                        continue

                    e_val = Emat[it, ie, ip, il]
                    a_val = az[it, ie, ip, il]
                    l_val = el[it, ie, ip, il]

                    # Boundary check for target grid
                    if e_val < e_edges[0] or e_val >= e_edges[-1]: continue
                    if a_val < az_edges[0] or a_val >= az_edges[-1]: continue
                    if l_val < el_edges[0] or l_val >= el_edges[-1]: continue

                    # Find target bin indices via binary search
                    idx_e = np.searchsorted(e_edges, e_val) - 1
                    idx_p = np.searchsorted(az_edges, a_val) - 1
                    idx_l = np.searchsorted(el_edges, l_val) - 1

                    # Accumulate counts and volume
                    res_accum[it, idx_e, idx_p, idx_l] += val
                    vol_accum[it, idx_e, idx_p, idx_l] += vol

    # Calculate Volume-Weighted Average
    # f_new = sum_counts / sum_vol
    psd_new = np.zeros((num_time, nE_new, nP_new, nT_new), dtype=np.float64)
    for it in prange(num_time):
        for ie in range(nE_new):
            for ip in range(nP_new):
                for il in range(nT_new):
                    if vol_accum[it, ie, ip, il] > 0:
                        psd_new[it, ie, ip, il] = res_accum[it, ie, ip, il] / vol_accum[it, ie, ip, il]

    return psd_new


def rebin_vdf(inp_psd, scpot=None, velocity_of_spacecraft=None,
              energy_new=None, phi_new=None, theta_new=None):
    """
    Rebins a 3D VDF into custom energy/angular bins with Numba acceleration.
    Uses volume-weighted averaging to ensure density consistency.
    """

    # 1. Set Default New Grids if not provided
    if energy_new is None:
        energy_new = 10 ** np.linspace(np.log10(1), np.log10(10000), 30)
    if phi_new is None:
        phi_new = np.linspace(0, 360, 18)
    if theta_new is None:
        theta_new = np.linspace(-90, 90, 10)

    # Helper to calculate bin edges from centers
    def get_edges(centers):
        centers = np.asanyarray(centers)
        dc = np.diff(centers)
        # Handle single bin case or uniform spacing
        if len(dc) == 0:
            return np.array([centers[0] - 0.5, centers[0] + 0.5])
        return np.concatenate(([centers[0] - dc[0] / 2], centers[:-1] + dc / 2, [centers[-1] + dc[-1] / 2]))

    e_edges = get_edges(energy_new)
    az_edges = get_edges(phi_new)
    el_edges = get_edges(theta_new)

    # 2. Extract Metadata
    time = inp_psd.time.data
    species = inp_psd.attrs["species"]
    psd_data = inp_psd.data.data
    energy_old = inp_psd.energy.data
    phi_old = inp_psd.phi.data
    theta_old = inp_psd.theta.data
    num_time, num_energy, num_phi, num_theta = psd_data.shape

    # 3. Expand Grids to 4D (Vectorized NumPy)
    logger.info("Expanding input grids...")
    res_old = expand_4d_grid(psd_data, energy_old, phi_old, theta_old, species)
    energymat = res_old["energymat"]
    phimat = res_old["phimat"]
    thetamat = res_old["thetamat"]
    d3v_old = res_old["d3v"]

    # 4. Spacecraft Potential Correction
    if scpot is not None:
        scpot_res = pyrf.resample(scpot, inp_psd.time).data
        energymat -= scpot_res[:, None, None, None]

    energymat = np.where(energymat > 0, energymat, np.nan)

    # 5. Transform to Cartesian Velocity (km/s)
    Vxmat, Vymat, Vzmat = vxyz_from_polar(energymat, phimat, thetamat, species)
    # Invert to look backward (standard instrument frame)
    Vxmat, Vymat, Vzmat = -Vxmat, -Vymat, -Vzmat

    # 6. Spacecraft Velocity (RAM) Correction
    if velocity_of_spacecraft is not None:
        v_sc = pyrf.resample(velocity_of_spacecraft, inp_psd.time).data
        Vxmat += v_sc[:, 0, None, None, None]
        Vymat += v_sc[:, 1, None, None, None]
        Vzmat += v_sc[:, 2, None, None, None]

    # 7-8. Re-compute Polar Coordinates in the New Frame
    Vmag = np.sqrt(Vxmat ** 2 + Vymat ** 2 + Vzmat ** 2)
    el = np.degrees(np.arctan2(Vzmat, np.sqrt(Vxmat ** 2 + Vymat ** 2)))
    az = np.degrees(np.arctan2(Vymat, Vxmat))
    az = np.mod(az, 360)  # Standardize to [0, 360]

    # 9. Recalculate Energy (eV)
    Emat = convert_energy_velocity(Vmag, "v2e", species)

    # 10-11. Core Rebinning with Volume-Weighted Numba kernel
    # Prep weighted PSD (f * d3v)
    psd_weighted = psd_data * d3v_old

    logger.info(f"Rebinning {num_time} time steps onto new grid {len(energy_new)}x{len(phi_new)}x{len(theta_new)}...")

    # The kernel now returns the normalized PSD directly
    psd_new = _rebin_kernel_numba(
        psd_weighted, d3v_old, Emat, az, el,
        e_edges, az_edges, el_edges,
        num_time, len(energy_new), len(phi_new), len(theta_new)
    )

    logger.info("Rebinning completed successfully.")

    # 12. Create Output Object
    return create_pdist_skymap(
        time, energy_new, psd_new, phi_new, theta_new,
        Units="s^3/m^6", species=species, direction_is_velocity=True
    )
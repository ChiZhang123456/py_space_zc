import numpy as np
import logging
import pyrfu.pyrf as pyrf

from .convert_energy_velocity import convert_energy_velocity
from .create_pdist_skymap import create_pdist_skymap
from .expand_4d_grid import expand_4d_grid
from .vxyz_from_polar import vxyz_from_polar

# Setup logging to display progress messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebin_vdf(inp_psd, scpot=None, velocity_of_spacecraft=None):
    """
    Rebins a 3D particle velocity distribution function (VDF) into new energy and angular bins.
    Applies spacecraft potential and velocity corrections, transforms data to velocity space,
    and redistributes phase space density accordingly.

    Parameters
    ----------
    inp_psd : ts_skymap
        Input 4D phase space density (PSD) in ts_skymap format with dimensions
        [time, energy, phi, theta].
    scpot : ts_scalar or None
        Spacecraft potential [V]. If None, defaults to 0.
    velocity_of_spacecraft : ts_vector or None
        Spacecraft velocity vector [km/s] in instrument coordinates. If None, defaults to 0.

    Returns
    -------
    vdf_rebin : ts_skymap
        Rebinned PSD in ts_skymap format using new energy, azimuthal, and elevation bins.
    """

    # 1. Define new target bin grids
    energy_new = 10 ** np.linspace(np.log10(1), np.log10(10000), 30)  # eV
    phi_new = np.linspace(0, 360 + 22.5, 18)                           # degrees, 0 to 382.5
    theta_new = np.linspace(-90, 90 + 22.5, 10)                        # degrees, -90 to 112.5
    bin_az = np.abs(np.mean(np.diff(phi_new)))                        # delta phi
    bin_el = np.abs(np.mean(np.diff(theta_new)))                      # delta theta
    dE = np.diff(energy_new, append=energy_new[-1])                   # delta energy

    # 2. Extract relevant fields from inp_psd
    time = inp_psd.time.data
    species = inp_psd.attrs["species"]
    psd_data = inp_psd.data.data                                      # shape: [t, e, phi, theta]
    energy_old = inp_psd.energy.data
    phi_old = inp_psd.phi.data
    theta_old = inp_psd.theta.data
    num_time, num_energy, num_phi, num_theta = psd_data.shape

    # 3. Expand into 4D grids of energy, phi, and theta for transformation
    res = expand_4d_grid(psd_data, energy_old, phi_old, theta_old, species)
    energymat = res["energymat"]       # shape: [t, e, phi, theta]
    phimat = res["phimat"]             # [0–360]
    thetamat = res["thetamat"]         # [0–180]

    # 4. Apply spacecraft potential correction
    if scpot is None:
        scpot = np.zeros(len(time))  # set to zero
    else:
        scpot = pyrf.resample(scpot, inp_psd.time).data              # align to psd time

    energymat -= scpot[:, None, None, None]           # subtract scpot
    energymat = np.where(energymat > 0, energymat, np.nan) # filter out unphysical energy

    # 5. Convert corrected (energy, phi, theta) to velocity vectors (km/s)
    Vxmat, Vymat, Vzmat = vxyz_from_polar(energymat, phimat, thetamat, species)

    # Reverse direction if needed (instrument frame often looks backward)
    Vxmat *= -1
    Vymat *= -1
    Vzmat *= -1

    # 6. Apply spacecraft velocity correction (in instrument coordinates)
    if velocity_of_spacecraft is None:
        velocity_of_spacecraft = np.zeros((len(time), 3))
    else:
        velocity_of_spacecraft = pyrf.resample(velocity_of_spacecraft, inp_psd.time).data

    vx_sc = np.tile(velocity_of_spacecraft[:, 0, None, None, None], [1, num_energy, num_phi, num_theta])
    vy_sc = np.tile(velocity_of_spacecraft[:, 1, None, None, None], [1, num_energy, num_phi, num_theta])
    vz_sc = np.tile(velocity_of_spacecraft[:, 2, None, None, None], [1, num_energy, num_phi, num_theta])
    Vxmat += vx_sc
    Vymat += vy_sc
    Vzmat += vz_sc

    # 7. Compute velocity magnitude and normalized direction
    Vmag = np.sqrt(Vxmat ** 2 + Vymat ** 2 + Vzmat ** 2)
    newX = Vxmat / Vmag
    newY = Vymat / Vmag
    newZ = Vzmat / Vmag

    # 8. Compute elevation angle [-90, 90] and azimuth angle [0, 360]
    Vxy = np.sqrt(newX ** 2 + newY ** 2)
    el = np.degrees(np.arctan2(newZ, Vxy))            # elevation
    az = np.degrees(np.arctan2(newY, newX))           # azimuth
    az[az < 0] += 360                                 # wrap azimuth to [0, 360]

    # 9. Recalculate energy based on corrected speed
    Emat = convert_energy_velocity(Vmag, "v2e", species)  # energy in eV

    # 10. Initialize output PSD array (rebinned result)
    psd_d3v_new = np.zeros((num_time, len(energy_new), len(phi_new), len(theta_new)))

    logger.info("Begin rebinning the VDF")

    # 11. Rebin PSD over new (energy, phi, theta) grid
    for it in range(num_time):
        temp_psd = psd_data[it]
        temp_en = Emat[it]
        temp_az = az[it]
        temp_el = el[it]

        for iE, E in enumerate(energy_new):
            for iAz, phi in enumerate(phi_new):
                for iEl, theta in enumerate(theta_new):
                    # Identify particles in current bin
                    mask = (
                        (temp_en >= E) & (temp_en < E + dE[iE]) &
                        (temp_az > phi - bin_az / 2) & (temp_az < phi + bin_az / 2) &
                        (temp_el > theta - bin_el / 2) & (temp_el < theta + bin_el / 2)
                    )
                    # Sum PSD values inside this bin (ignore NaNs)
                    psd_d3v_new[it, iE, iAz, iEl] = np.nansum(temp_psd[mask])

    logger.info("Finish the rebinning")

    # 12. Return result as ts_skymap
    vdf_rebin = create_pdist_skymap(
        time,
        energy_new,
        psd_d3v_new,
        phi_new,
        theta_new,
        Units="s^3/m^6",          # Units of phase space density
        species=species,
        direction_is_velocity=True,
    )

    return vdf_rebin

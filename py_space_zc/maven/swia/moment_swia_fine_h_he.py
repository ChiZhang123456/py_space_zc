import numpy as np
from scipy import constants

from py_space_zc import ts_scalar, ts_vec_xyz


_SWIA_IDL_PROTON_MASS = 5.68566e-06 * 1836.0
_ERG_PER_EV = 1.6e-12
_KG_PER_IDL_MASS = 1.6e-22


def _as_numpy(obj, name):
    if isinstance(obj, dict):
        return np.asarray(obj[name])
    if name == "DEF":
        return np.asarray(obj.data.data)
    if name == "time":
        return np.asarray(obj.time.data)
    return np.asarray(getattr(obj, name).data)


def _get_geometry(swia_3d):
    if isinstance(swia_3d, dict):
        energy = np.asarray(swia_3d["energy"], dtype=float)
        theta = np.asarray(swia_3d["theta"], dtype=float)
        dtheta = np.asarray(swia_3d["dtheta"], dtype=float)
        phi = np.asarray(swia_3d["phi"], dtype=float)
        dphi = np.asarray(swia_3d.get("dphi", np.full(phi.shape, 4.5)), dtype=float)
        de_over_e = float(swia_3d.get("de_over_e", 0.075))
        return energy, theta, dtheta, phi, dphi, de_over_e

    attrs = swia_3d.attrs
    energy = np.asarray(swia_3d.energy.data, dtype=float)
    theta = np.asarray(attrs.get("swia_theta_elevation"), dtype=float)
    dtheta = np.asarray(attrs.get("swia_dtheta_elevation"), dtype=float)
    phi = np.asarray(attrs.get("swia_phi_elevation_frame"), dtype=float)
    dphi = np.asarray(attrs.get("swia_dphi", np.full(phi.shape, 4.5)), dtype=float)
    de_over_e = float(attrs.get("swia_de_over_e", 0.075))

    if theta.size == 1 or dtheta.size == 1 or phi.size == 1:
        raise ValueError(
            "SWIA fine geometry is missing. Load data with "
            "maven.get_data(..., 'swia_3d_fine') or pass read_swia_3d output."
        )

    return energy, theta, dtheta, phi, dphi, de_over_e


def _broadcast_geometry(data, energy, theta, dtheta, phi, dphi, de_over_e):
    nt, ne, nphi, nth = data.shape

    if energy.ndim == 1:
        energy = np.broadcast_to(energy[None, :], (nt, ne))
    if theta.ndim == 2:
        theta = np.broadcast_to(theta[None, :, :], (nt, ne, nth))
    if dtheta.ndim == 2:
        dtheta = np.broadcast_to(dtheta[None, :, :], (nt, ne, nth))

    denergy = energy * de_over_e

    energy4 = energy[:, :, None, None]
    denergy4 = denergy[:, :, None, None]
    theta4 = theta[:, :, None, :]
    dtheta4 = dtheta[:, :, None, :]
    phi4 = phi[None, None, :, None]
    dphi4 = dphi[None, None, :, None]

    return energy4, denergy4, theta4, dtheta4, phi4, dphi4


def _solid_angle(theta_deg, dtheta_deg, dphi_deg):
    theta = np.deg2rad(theta_deg)
    dtheta = np.deg2rad(dtheta_deg)
    dphi = np.deg2rad(dphi_deg)
    return 2.0 * dphi * np.cos(theta) * np.sin(0.5 * dtheta)


def _moments_from_def(data, energy, denergy, theta, dtheta, phi, dphi,
                      mass_idl=_SWIA_IDL_PROTON_MASS):
    data = np.asarray(data, dtype=float)
    data = np.where(np.isfinite(data) & (data > 0.0), data, 0.0)

    domega = _solid_angle(theta, dtheta, dphi)
    mass = mass_idl * _KG_PER_IDL_MASS
    energy2 = energy[:, :, 0, 0]
    denergy2 = denergy[:, :, 0, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        density_const = np.sqrt(mass / (2.0 * _ERG_PER_EV))
        density = density_const * np.nansum(
            denergy2 * energy2 ** (-1.5) * np.nansum(data * domega, axis=(2, 3)),
            axis=1,
        )

        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        cth = np.cos(theta_rad)
        sth = np.sin(theta_rad)
        cph = np.cos(phi_rad)
        sph = np.sin(phi_rad)

        sumx = np.nansum(data * cph * domega * cth, axis=(2, 3))
        sumy = np.nansum(data * sph * domega * cth, axis=(2, 3))
        sumz = np.nansum(data * domega * sth, axis=(2, 3))

        flux = np.stack(
            [
                np.nansum(denergy2 * energy2 ** (-1.0) * sumx, axis=1),
                np.nansum(denergy2 * energy2 ** (-1.0) * sumy, axis=1),
                np.nansum(denergy2 * energy2 ** (-1.0) * sumz, axis=1),
            ],
            axis=1,
        )

        vel_cm_s = flux / density[:, None]
        velocity = 1.0e-5 * vel_cm_s

        pressure_const = (mass / (2.0 * _ERG_PER_EV)) ** (-0.5)
        sumxx = np.nansum(data * cph * cph * domega * cth * cth, axis=(2, 3))
        sumyy = np.nansum(data * sph * sph * domega * cth * cth, axis=(2, 3))
        sumzz = np.nansum(data * domega * sth * sth, axis=(2, 3))
        sumxy = np.nansum(data * cph * sph * domega * cth * cth, axis=(2, 3))
        sumxz = np.nansum(data * cph * domega * cth * sth, axis=(2, 3))
        sumyz = np.nansum(data * sph * domega * cth * sth, axis=(2, 3))

        dnrg_p = pressure_const * denergy2 * energy2 ** (-0.5)
        p_raw = np.stack(
            [
                np.nansum(dnrg_p * sumxx, axis=1),
                np.nansum(dnrg_p * sumyy, axis=1),
                np.nansum(dnrg_p * sumzz, axis=1),
                np.nansum(dnrg_p * sumxy, axis=1),
                np.nansum(dnrg_p * sumxz, axis=1),
                np.nansum(dnrg_p * sumyz, axis=1),
            ],
            axis=1,
        )

        pressure = np.empty_like(p_raw)
        pressure[:, 0] = mass * (p_raw[:, 0] - vel_cm_s[:, 0] * flux[:, 0]) / _ERG_PER_EV
        pressure[:, 1] = mass * (p_raw[:, 1] - vel_cm_s[:, 1] * flux[:, 1]) / _ERG_PER_EV
        pressure[:, 2] = mass * (p_raw[:, 2] - vel_cm_s[:, 2] * flux[:, 2]) / _ERG_PER_EV
        pressure[:, 3] = mass * (p_raw[:, 3] - vel_cm_s[:, 0] * flux[:, 1]) / _ERG_PER_EV
        pressure[:, 4] = mass * (p_raw[:, 4] - vel_cm_s[:, 0] * flux[:, 2]) / _ERG_PER_EV
        pressure[:, 5] = mass * (p_raw[:, 5] - vel_cm_s[:, 1] * flux[:, 2]) / _ERG_PER_EV

        temperature = (pressure[:, 0] + pressure[:, 1] + pressure[:, 2]) / (3.0 * density)

    bad = ~np.isfinite(density) | (density <= 0.0)
    density = np.where(bad, np.nan, density)
    velocity[bad, :] = np.nan
    temperature = np.where(bad, np.nan, temperature)

    return density, velocity, temperature, pressure


def _estimate_energy_cut(data, energy, denergy, theta, dtheta, phi, dphi):
    n0, v0, _, _ = _moments_from_def(data, energy, denergy, theta, dtheta, phi, dphi)
    speed2 = np.nansum(v0 * v0, axis=1)
    e0 = 0.5 * constants.proton_mass * speed2 * 1.0e6 / constants.elementary_charge

    domega = _solid_angle(theta, dtheta, dphi)
    espec = np.nansum(data * domega, axis=(2, 3)) / np.nansum(domega, axis=(2, 3))
    e1 = energy[:, :, 0, 0]

    ecut = 1.5 * e0
    for i in range(data.shape[0]):
        if not np.isfinite(e0[i]) or e0[i] <= 0.0:
            continue
        w = np.where((e1[i] > 1.15 * e0[i]) & (e1[i] < 2.0 * e0[i]))[0]
        if w.size > 2:
            grad = np.gradient(espec[i, w], e1[i, w])
            turns = np.where((grad[:-1] < 0.0) & (grad[1:] >= 0.0))[0]
            if turns.size:
                ecut[i] = e1[i, w[turns[0] + 1]]
            else:
                ecut[i] = e1[i, w[np.nanargmin(espec[i, w])]]
    return ecut


def moment_swia_fine_h_he(swia_3d, rotate_to_mso=True):
    """
    Compute H+ and He++ density, velocity, and scalar temperature from SWIA fine 3D.

    The algorithm follows the SPEDAS IDL routine
    ``mvn_swia_protonalphamom_minf``: first compute a proton moment from the
    full fine distribution, estimate the proton bulk energy, find the minimum
    between the proton and alpha peaks, then integrate the low-energy and
    high-energy parts separately. The He++ part uses the same measured
    differential energy flux, with mass multiplied by 4 and energy by 2.
    """
    time = _as_numpy(swia_3d, "time")
    data = _as_numpy(swia_3d, "DEF")
    energy, theta, dtheta, phi, dphi, de_over_e = _get_geometry(swia_3d)
    energy4, denergy4, theta4, dtheta4, phi4, dphi4 = _broadcast_geometry(
        data, energy, theta, dtheta, phi, dphi, de_over_e
    )

    ecut = _estimate_energy_cut(data, energy4, denergy4, theta4, dtheta4, phi4, dphi4)

    proton_data = np.where(energy4 <= ecut[:, None, None, None], data, 0.0)
    alpha_data = np.where(energy4 >= ecut[:, None, None, None], data, 0.0)

    n_h, v_h, t_h, p_h = _moments_from_def(
        proton_data, energy4, denergy4, theta4, dtheta4, phi4, dphi4
    )
    n_he, v_he, t_he, p_he = _moments_from_def(
        alpha_data, 2.0 * energy4, 2.0 * denergy4, theta4, dtheta4, phi4, dphi4,
        mass_idl=4.0 * _SWIA_IDL_PROTON_MASS,
    )

    coords = "SWIA"
    if rotate_to_mso:
        from py_space_zc import maven

        v_h_da = ts_vec_xyz(time, v_h)
        v_he_da = ts_vec_xyz(time, v_he)
        v_h = maven.coords_convert(v_h_da, "swia2mso").data
        v_he = maven.coords_convert(v_he_da, "swia2mso").data
        coords = "MSO"

    return {
        "nH": ts_scalar(time, n_h, attrs={"units": "cm^-3", "species": "H+"}),
        "VH": ts_vec_xyz(time, v_h, attrs={"units": "km/s", "coordinates": coords, "species": "H+"}),
        "TH": ts_scalar(time, t_h, attrs={"units": "eV", "species": "H+"}),
        "nHe": ts_scalar(time, n_he, attrs={"units": "cm^-3", "species": "He++"}),
        "VHe": ts_vec_xyz(time, v_he, attrs={"units": "km/s", "coordinates": coords, "species": "He++"}),
        "THe": ts_scalar(time, t_he, attrs={"units": "eV", "species": "He++"}),
        "ecut": ts_scalar(time, ecut, attrs={"units": "eV"}),
        "PH": p_h,
        "PHe": p_he,
    }

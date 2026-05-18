import numpy as np

from .cex_cross_section import cex_cross_section


_REACTION_MAP = {
    "hpo>oph": ("H+", "O"),
    "hph>hph": ("H+", "H"),
    "oph>hpo": ("O+", "H"),
    "opo>opo": ("O+", "O"),
    "hpn2>n2ph": ("H+", "N2"),
    "hpo2>o2ph": ("H+", "O2"),
}


def _reaction_to_species(reaction_type):
    key = str(reaction_type).strip().lower()
    if key not in _REACTION_MAP:
        raise ValueError(
            "Unsupported charge-exchange reaction type: "
            f"{reaction_type!r}. Supported types are: "
            + ", ".join(sorted(_REACTION_MAP))
        )
    return _REACTION_MAP[key]


def _energy_bin_width(energy):
    """Return positive energy-bin widths in eV for ascending or descending bins."""
    if energy.size == 1:
        raise ValueError("energy must contain at least two bins to estimate dE.")

    dE = np.abs(np.diff(energy))
    dE = np.concatenate([dE, dE[-1:]])
    return dE


def cex_prod_rate(
    energy,
    ion_DEF,
    reaction_type=None,
    ion=None,
    neutral=None,
    dtheta=np.pi / 2,
    dphi=2 * np.pi,
    max_energy_eV=4000.0,
):
    """
    Compute charge-exchange production rate from ion differential energy flux.

    This is the Python counterpart of the MATLAB
    ``charge_exchange_prod_rate.m`` workflow. The calculation uses

        rate = integral(DEF / E * sigma_cex(E) dE dOmega)

    where ``DEF / E`` converts differential energy flux to differential
    particle flux.

    Parameters
    ----------
    energy : array-like, shape (n_energy,)
        Ion energy bins in eV. Both ascending and descending energy order are
        accepted.
    ion_DEF : array-like, shape (n_time, n_energy) or (n_energy,)
        Differential energy flux in eV / (cm^2 s sr eV). A 1-D input is treated
        as one time sample.
    reaction_type : str, optional
        MATLAB-style reaction name, for example ``'HpO>OpH'``,
        ``'HpH>HpH'``, ``'OpH>HpO'``, or ``'OpO>OpO'``. If supplied, this is
        used to infer ``ion`` and ``neutral``.
    ion : str, optional
        Incident ion species passed to ``cex_cross_section``, for example
        ``'H+'`` or ``'O+'``. Required if ``reaction_type`` is not supplied.
    neutral : str, optional
        Neutral target species passed to ``cex_cross_section``, for example
        ``'H'``, ``'O'``, ``'N2'``, or ``'O2'``. Required if
        ``reaction_type`` is not supplied.
    dtheta : float, optional
        Integrated polar-angle width in radians. Default is pi/2, matching the
        MATLAB routine.
    dphi : float, optional
        Integrated azimuthal-angle width in radians. Default is 2*pi.
    max_energy_eV : float or None, optional
        Energies greater than or equal to this value are ignored. The default
        is 4000 eV, matching ``charge_exchange_prod_rate.m``. Set to None to
        use all energies.

    Returns
    -------
    prod_rate : ndarray or float
        Charge-exchange production rate in s^-1. A scalar is returned when
        ``ion_DEF`` is 1-D, otherwise the shape is (n_time,).

    Notes
    -----
    Units:
      - ``ion_DEF``: eV / (cm^2 s sr eV)
      - ``energy`` and ``dE``: eV
      - ``cex_cross_section``: cm^2
      - ``dOmega``: sr
      - result: s^-1
    """
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 1:
        raise ValueError("energy must be a 1-D array.")
    if np.any(energy <= 0):
        raise ValueError("energy must be positive because DEF is divided by E.")

    ion_DEF = np.asarray(ion_DEF, dtype=float)
    scalar_time = ion_DEF.ndim == 1
    if scalar_time:
        ion_DEF = ion_DEF[None, :]
    elif ion_DEF.ndim != 2:
        raise ValueError("ion_DEF must be a 1-D or 2-D array.")

    if ion_DEF.shape[1] != energy.size:
        raise ValueError(
            "ion_DEF must have the same number of energy bins as energy. "
            f"Got ion_DEF.shape[1]={ion_DEF.shape[1]} and energy.size={energy.size}."
        )

    if reaction_type is not None:
        ion_from_type, neutral_from_type = _reaction_to_species(reaction_type)
        ion = ion_from_type if ion is None else ion
        neutral = neutral_from_type if neutral is None else neutral

    if ion is None or neutral is None:
        raise ValueError("Specify either reaction_type or both ion and neutral.")

    energy_mat = energy[None, :]
    dE = _energy_bin_width(energy)[None, :]
    domega = dtheta * dphi

    dpf = ion_DEF / energy_mat
    if max_energy_eV is not None:
        dpf = np.where(energy_mat >= max_energy_eV, np.nan, dpf)

    sigma = np.asarray(cex_cross_section(energy, ion=ion, neutral=neutral), dtype=float)
    if sigma.shape != energy.shape:
        sigma = np.broadcast_to(sigma, energy.shape)

    prod_rate = np.nansum(dpf * sigma[None, :] * domega * dE, axis=1)
    if scalar_time:
        return float(prod_rate[0])
    return prod_rate


charge_exchange_prod_rate = cex_prod_rate


__all__ = ["cex_prod_rate", "charge_exchange_prod_rate"]

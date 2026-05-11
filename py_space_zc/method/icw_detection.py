"""
Pickup ion cyclotron wave detection from MAVEN magnetic field data.

The default criteria follow the event-selection logic used in ICW studies:
mean-field-aligned coordinates, Welch spectral matrix, transverse-dominant
power near the local ion gyrofrequency, left-handed ellipticity, coherent
polarization, and quasi-parallel wave normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np


E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


ION_TABLE = {
    "H+": (1.007276, 1.0),
    "H": (1.007276, 1.0),
    "He+": (4.002602, 1.0),
    "He": (4.002602, 1.0),
    "O+": (15.999, 1.0),
    "O": (15.999, 1.0),
    "O2+": (31.998, 1.0),
    "O2": (31.998, 1.0),
    "CO2+": (44.0095, 1.0),
    "CO2": (44.0095, 1.0),
}


@dataclass
class ICWCriteria:
    transverse_ratio_min: float = 5.0
    ellipticity_max: float = -0.55
    dop_min: float = 0.7
    wave_angle_max_deg: float = 35.0
    gyro_band_low_factor: float = 0.8
    gyro_band_high_factor: float = 1.2
    neighbor_band_low_factor: float = 0.6
    neighbor_band_high_factor: float = 1.4
    require_peak_in_band: bool = True
    peak_neighbor_prominence_min: float = 3.0
    require_peak_above_both_sides: bool = True
    require_narrowband_peak: bool = True
    narrowband_max_width_factor: float = 0.65
    narrowband_min_peak_to_band_median: float = 1.5
    require_perp_peak_stronger_than_para: bool = True
    perp_para_peak_prominence_ratio_min: float = 1.5
    require_sharp_perp_peak: bool = True
    sharp_peak_adjacent_ratio_min: float = 1.6


@dataclass
class PCWSVDCriteria:
    period_range_s: tuple[float, float] = (1.0, 50.0)
    gyro_band_low_factor: float = 0.7
    gyro_band_high_factor: float = 1.3
    transverse_ratio_min: float = 1.5
    ellipticity_max: float = -0.50
    wave_angle_max_deg: float = 45.0
    planarity_min: float = 0.45
    narrowband_prominence_min: float = 2.0
    median_narrowband_prominence_min: float = 8.0
    perp_para_peak_prominence_ratio_min: float = 1.3
    min_good_time_fraction: float = 0.50
    min_continuous_duration_s: float = 120.0
    min_good_windows: int = 3


def _as_seconds(time: np.ndarray) -> np.ndarray:
    """Convert numeric or datetime64 time to seconds from the first sample."""
    time = np.asarray(time)
    if np.issubdtype(time.dtype, np.datetime64):
        return (time - time[0]) / np.timedelta64(1, "s")
    return time.astype(float) - float(time[0])


def _sampling_rate_hz(time: np.ndarray) -> float:
    tsec = _as_seconds(time)
    dt = np.nanmedian(np.diff(tsec))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time must be strictly increasing with a valid cadence")
    return 1.0 / dt


def _mfa_basis(b_mean: np.ndarray, reference_axis: np.ndarray | None = None) -> np.ndarray:
    """Return a 3 by 3 matrix whose rows are e_perp1, e_perp2, e_parallel."""
    bnorm = np.linalg.norm(b_mean)
    if not np.isfinite(bnorm) or bnorm == 0:
        raise ValueError("mean magnetic field is zero or invalid")

    epar = b_mean / bnorm
    ref = np.array([1.0, 0.0, 0.0]) if reference_axis is None else np.asarray(reference_axis, dtype=float)
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        raise ValueError("reference_axis must be nonzero")
    ref = ref / ref_norm

    eperp2 = np.cross(epar, ref)
    if np.linalg.norm(eperp2) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0])
        eperp2 = np.cross(epar, ref)
    eperp2 = eperp2 / np.linalg.norm(eperp2)
    eperp1 = np.cross(eperp2, epar)
    eperp1 = eperp1 / np.linalg.norm(eperp1)
    return np.vstack((eperp1, eperp2, epar))


def _ion_mass_charge(ion: str | tuple[float, float]) -> tuple[float, float]:
    """
    Return ion mass in kg and charge in C.

    For a custom ion, pass (mass_amu, charge_state), for example (16, 1).
    """
    if isinstance(ion, tuple):
        mass_amu, charge_state = ion
    else:
        key = ion.strip()
        if key not in ION_TABLE:
            raise ValueError(f"unknown ion {ion!r}; use one of {sorted(ION_TABLE)} or pass (mass_amu, charge_state)")
        mass_amu, charge_state = ION_TABLE[key]
    return float(mass_amu) * AMU, float(charge_state) * E_CHARGE


def _welch_spectral_matrix(
    b_mfa: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 3 by 3 magnetic spectral matrix using Welch averaging."""
    n = b_mfa.shape[0]
    if nperseg is None:
        nperseg = min(512, n)
    nperseg = int(min(nperseg, n))
    if nperseg < 8:
        raise ValueError("not enough samples for spectral analysis")

    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = np.arange(0, n - nperseg + 1, step)
    if starts.size == 0:
        starts = np.array([0])

    window = np.hanning(nperseg)
    scale = fs * np.sum(window ** 2)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    smat = np.zeros((freqs.size, 3, 3), dtype=complex)

    for start in starts:
        seg = b_mfa[start : start + nperseg] - np.nanmean(b_mfa[start : start + nperseg], axis=0)
        seg = np.where(np.isfinite(seg), seg, 0.0)
        fft = np.fft.rfft(seg * window[:, None], axis=0)
        for i in range(3):
            for j in range(3):
                smat[:, i, j] += fft[:, i] * np.conj(fft[:, j]) / scale

    smat /= starts.size
    if freqs.size > 2:
        smat[1:-1] *= 2.0
    return freqs, smat


def _polarization_parameters(smat: np.ndarray) -> dict[str, np.ndarray]:
    s11 = np.real(smat[:, 0, 0])
    s22 = np.real(smat[:, 1, 1])
    s33 = np.real(smat[:, 2, 2])

    p_perp = 0.5 * (s11 + s22)
    p_par = s33

    n_freq = smat.shape[0]
    ellipticity = np.full(n_freq, np.nan)
    planarity = np.full(n_freq, np.nan)
    wave_angle = np.full(n_freq, np.nan)
    normal = np.full((n_freq, 3), np.nan)

    for i in range(n_freq):
        if not np.all(np.isfinite(smat[i])):
            continue
        a_mat = np.vstack((np.real(smat[i].T), -np.imag(smat[i].T)))
        try:
            _, singular_values, vh = np.linalg.svd(a_mat, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if singular_values[0] <= 0:
            continue

        ellipticity[i] = singular_values[1] / singular_values[0] * np.sign(np.imag(smat[i, 0, 1]))
        planarity[i] = 1.0 - np.sqrt(singular_values[2] / singular_values[0])

        k_vec = np.array([vh[0, 2], vh[1, 2], vh[2, 2]], dtype=float)
        sign_kz = np.sign(k_vec[2])
        if sign_kz == 0:
            sign_kz = 1.0
        k_vec *= sign_kz
        normal[i] = k_vec
        wave_angle[i] = np.degrees(np.abs(np.arctan2(np.hypot(k_vec[0], k_vec[1]), k_vec[2])))

    with np.errstate(invalid="ignore", divide="ignore"):
        trace = np.trace(smat, axis1=1, axis2=2)
        dop = np.sqrt(
            np.real(
                1.5
                * np.trace(np.matmul(smat, smat), axis1=1, axis2=2)
                / trace ** 2
                - 0.5
            )
        )

    circular_polarization = np.sign(ellipticity) * np.abs(ellipticity)

    return {
        "p_perp": p_perp,
        "p_par": p_par,
        "ellipticity": ellipticity,
        "circular_polarization": circular_polarization,
        "dop": dop,
        "wave_angle_deg": wave_angle,
        "planarity": planarity,
        "wave_normal_fac": normal,
    }


def _is_xarray_like(obj: Any) -> bool:
    return hasattr(obj, "dims") and hasattr(obj, "coords")


def _coord_values(obj: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if hasattr(obj, "coords") and name in obj.coords:
            return np.asarray(obj.coords[name].values)
        if hasattr(obj, name):
            val = getattr(obj, name)
            if hasattr(val, "values"):
                return np.asarray(val.values)
            return np.asarray(val)
    return None


def _extract_bwave_arrays(
    bwave: Any,
    *,
    time_name: str | None = None,
    b_var: str | None = None,
    component_dim: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time and Nx3 magnetic field from an xarray DataArray or Dataset."""
    if not _is_xarray_like(bwave):
        raise TypeError("Bwave must be an xarray DataArray or Dataset")

    time_candidates = (time_name,) if time_name is not None else ("time", "Time", "epoch", "Epoch", "datetime", "t")
    time_candidates = tuple(x for x in time_candidates if x is not None)
    time = _coord_values(bwave, time_candidates)

    data_obj = None
    if hasattr(bwave, "data_vars"):
        if b_var is not None:
            data_obj = bwave[b_var]
        else:
            preferred = ("Bwave", "B", "b", "B_MSO", "B_mso", "Bvec", "mag")
            for name in preferred:
                if name in bwave.data_vars:
                    data_obj = bwave[name]
                    break
            if data_obj is None:
                for name in bwave.data_vars:
                    candidate = bwave[name]
                    if np.asarray(candidate.values).ndim == 2 and 3 in np.asarray(candidate.values).shape:
                        data_obj = candidate
                        break
    else:
        data_obj = bwave

    if data_obj is None:
        raise ValueError("cannot find a 3-component magnetic field variable in Bwave")

    data = np.asarray(data_obj.values, dtype=float)
    dims = tuple(getattr(data_obj, "dims", ()))
    if data.ndim != 2 or 3 not in data.shape:
        raise ValueError("Bwave magnetic field data must be a 2D array with one dimension of length 3")

    if time is None:
        time = _coord_values(data_obj, time_candidates)
    if time is None:
        time_dim = next((d for d in dims if data_obj.sizes.get(d, None) != 3), None)
        if time_dim is not None and time_dim in data_obj.coords:
            time = np.asarray(data_obj.coords[time_dim].values)
    if time is None:
        raise ValueError("cannot find time coordinate in Bwave")

    comp_axis = None
    if component_dim is not None and component_dim in dims:
        comp_axis = dims.index(component_dim)
    elif data.shape[1] == 3:
        comp_axis = 1
    elif data.shape[0] == 3:
        comp_axis = 0
    if comp_axis is None:
        raise ValueError("cannot identify the 3-component dimension in Bwave")

    if comp_axis == 0:
        data = data.T
    if len(time) != data.shape[0]:
        raise ValueError("time coordinate length does not match Bwave sample length")
    return np.asarray(time), data


def _call_psd_welch(bwave: Any, psd_kwargs: dict[str, Any] | None) -> Any:
    try:
        from py_space_zc import method
    except ImportError as exc:
        raise ImportError("py_space_zc is not available, cannot call py_space_zc.method.psd_welch") from exc
    kwargs = {"plot": False}
    if psd_kwargs is not None:
        kwargs.update(psd_kwargs)
    return method.psd_welch(bwave, **kwargs)


def _available_fields(obj: Any) -> list[str]:
    if hasattr(obj, "data_vars"):
        return list(obj.data_vars) + [x for x in obj.coords if x not in obj.data_vars]
    if isinstance(obj, Mapping):
        return list(obj.keys())
    if hasattr(obj, "_fields"):
        return list(obj._fields)
    return [x for x in dir(obj) if not x.startswith("_")]


def _get_field(obj: Any, names: tuple[str, ...]) -> Any | None:
    lower_names = tuple(x.lower() for x in names)
    fields = _available_fields(obj)
    for field in fields:
        if field.lower() in lower_names:
            return obj[field] if isinstance(obj, Mapping) or hasattr(obj, "coords") else getattr(obj, field)
    for field in fields:
        low = field.lower()
        if any(name in low for name in lower_names):
            return obj[field] if isinstance(obj, Mapping) or hasattr(obj, "coords") else getattr(obj, field)
    return None


def _values(obj: Any) -> np.ndarray:
    return np.asarray(obj.values if hasattr(obj, "values") else obj, dtype=float)


def _extract_psd_welch_products(psd: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract frequency, perpendicular PSD, and parallel PSD from psd_welch output.

    The helper accepts common xarray Dataset/DataArray, dict, and object outputs.
    If your local psd_welch uses different names, pass its output once and the
    error message will list the visible fields.
    """
    if isinstance(psd, Mapping) and {"f", "psd_para", "psd_perp1", "psd_perp2"}.issubset(psd):
        freq = np.asarray(psd["f"], dtype=float)
        ppar = np.asarray(psd["psd_para"], dtype=float)
        pperp = np.asarray(psd["psd_perp1"], dtype=float) + np.asarray(psd["psd_perp2"], dtype=float)
        return freq, pperp, ppar

    if hasattr(psd, "coords") and not hasattr(psd, "data_vars"):
        values = np.asarray(psd.values, dtype=float)
        if values.ndim == 2:
            for dim in psd.dims:
                if dim in psd.coords:
                    labels = np.asarray(psd.coords[dim].values)
                    labels_low = np.asarray([str(x).lower() for x in labels])
                    perp_hits = np.where(np.char.find(labels_low, "perp") >= 0)[0]
                    para_hits = np.where((np.char.find(labels_low, "para") >= 0) | (np.char.find(labels_low, "par") >= 0))[0]
                    if perp_hits.size and para_hits.size:
                        comp_axis = psd.dims.index(dim)
                        freq_axis = 1 - comp_axis
                        freq_dim = psd.dims[freq_axis]
                        if freq_dim in psd.coords:
                            freq = np.asarray(psd.coords[freq_dim].values, dtype=float)
                            pperp = np.take(values, int(perp_hits[0]), axis=comp_axis)
                            ppar = np.take(values, int(para_hits[0]), axis=comp_axis)
                            return freq, np.asarray(pperp, dtype=float), np.asarray(ppar, dtype=float)

    freq_obj = _get_field(psd, ("freq", "freqs", "frequency", "f"))
    pperp_obj = _get_field(psd, ("p_perp", "pperp", "perp", "psd_perp", "psdperp", "transverse"))
    ppar_obj = _get_field(psd, ("p_par", "ppar", "para", "parallel", "psd_para", "psdpar", "compressional"))

    if pperp_obj is None or ppar_obj is None:
        fields = ", ".join(_available_fields(psd))
        raise ValueError(f"cannot find perp/para PSD in psd_welch output. Available fields: {fields}")

    if freq_obj is None:
        for obj in (pperp_obj, ppar_obj):
            if hasattr(obj, "coords"):
                for name in ("freq", "freqs", "frequency", "f"):
                    if name in obj.coords:
                        freq_obj = obj.coords[name]
                        break
            if freq_obj is not None:
                break
    if freq_obj is None:
        fields = ", ".join(_available_fields(psd))
        raise ValueError(f"cannot find frequency coordinate in psd_welch output. Available fields: {fields}")

    return _values(freq_obj), _values(pperp_obj), _values(ppar_obj)


def _integrate_band(y: np.ndarray, x: np.ndarray, band: np.ndarray) -> float:
    if np.count_nonzero(band) > 1:
        return float(np.trapz(y[band], x[band]))
    return float(y[band][0])


def _band_average(y: np.ndarray, band: np.ndarray) -> float:
    finite = band & np.isfinite(y)
    if not np.any(finite):
        return np.nan
    return float(np.nanmedian(y[finite]))


def _pcw_neighbor_peak_metrics(
    freq: np.ndarray,
    p_perp: np.ndarray,
    p_para: np.ndarray,
    center_band: np.ndarray,
    low_neighbor_band: np.ndarray,
    high_neighbor_band: np.ndarray,
    prominence_min: float,
    narrowband_max_width_factor: float,
    narrowband_min_peak_to_band_median: float,
    perp_para_peak_prominence_ratio_min: float,
    sharp_peak_adjacent_ratio_min: float,
) -> dict[str, Any]:
    """Check whether transverse PSD is enhanced relative to adjacent bands."""
    if not np.any(center_band):
        raise ValueError("center_band must contain at least one frequency bin")

    band_idx = np.flatnonzero(center_band)
    local_idx = band_idx[int(np.nanargmax(p_perp[center_band]))]
    peak_power = float(p_perp[local_idx])
    peak_frequency = float(freq[local_idx])

    finite_power = np.asarray(p_perp, dtype=float)
    finite_para = np.asarray(p_para, dtype=float)
    center_power = _band_average(finite_power, center_band)
    low_neighbor_power = _band_average(p_perp, low_neighbor_band)
    high_neighbor_power = _band_average(p_perp, high_neighbor_band)
    if not np.isfinite(low_neighbor_power):
        before = np.flatnonzero(freq < freq[band_idx[0]])
        if before.size:
            low_neighbor_power = float(p_perp[before[-1]])
    if not np.isfinite(high_neighbor_power):
        after = np.flatnonzero(freq > freq[band_idx[-1]])
        if after.size:
            high_neighbor_power = float(p_perp[after[0]])
    background = np.nanmax([low_neighbor_power, high_neighbor_power])

    left_search = np.flatnonzero((freq >= 0.5 * freq[band_idx[0]]) & (freq < freq[local_idx]))
    right_search = np.flatnonzero((freq > freq[local_idx]) & (freq <= 2.0 * freq[band_idx[-1]]))
    left_floor = float(np.nanmin(finite_power[left_search])) if left_search.size else np.nan
    right_floor = float(np.nanmin(finite_power[right_search])) if right_search.size else np.nan
    side_floor = np.nanmax([left_floor, right_floor])

    if np.isfinite(background) and background > 0:
        prominence = peak_power / background
    else:
        prominence = np.inf if peak_power > 0 else np.nan
    if np.isfinite(side_floor) and side_floor > 0:
        side_prominence = peak_power / side_floor
    else:
        side_prominence = np.inf if peak_power > 0 else np.nan

    center_median_power = _band_average(finite_power, center_band)
    if np.isfinite(center_median_power) and center_median_power > 0:
        peak_to_band_median = peak_power / center_median_power
    else:
        peak_to_band_median = np.inf if peak_power > 0 else np.nan

    if 0 < local_idx < freq.size - 1:
        is_local_max = bool(peak_power > p_perp[local_idx - 1] and peak_power > p_perp[local_idx + 1])
        adjacent_background = float(np.nanmax([p_perp[local_idx - 1], p_perp[local_idx + 1]]))
    else:
        is_local_max = False
        adjacent_background = np.nan
    if np.isfinite(adjacent_background) and adjacent_background > 0:
        sharp_peak_adjacent_ratio = peak_power / adjacent_background
    else:
        sharp_peak_adjacent_ratio = np.inf if peak_power > 0 else np.nan

    half_prominence_level = np.nan
    half_prominence_width_hz = np.nan
    half_prominence_width_factor = np.nan
    if np.isfinite(side_floor) and np.isfinite(peak_power) and peak_power > side_floor:
        half_prominence_level = side_floor + 0.5 * (peak_power - side_floor)
        left = local_idx
        while left > 0 and p_perp[left - 1] >= half_prominence_level:
            left -= 1
        right = local_idx
        while right < freq.size - 1 and p_perp[right + 1] >= half_prominence_level:
            right += 1
        half_prominence_width_hz = float(freq[right] - freq[left])
        if peak_frequency > 0:
            half_prominence_width_factor = half_prominence_width_hz / peak_frequency

    has_narrowband_peak = bool(
        np.isfinite(half_prominence_width_factor)
        and half_prominence_width_factor <= narrowband_max_width_factor
        and np.isfinite(peak_to_band_median)
        and peak_to_band_median >= narrowband_min_peak_to_band_median
    )
    has_sharp_peak = bool(
        np.isfinite(sharp_peak_adjacent_ratio)
        and sharp_peak_adjacent_ratio >= sharp_peak_adjacent_ratio_min
    )

    has_two_sided_peak = bool(
        is_local_max
        and np.isfinite(left_floor)
        and np.isfinite(right_floor)
        and peak_power >= prominence_min * left_floor
        and peak_power >= prominence_min * right_floor
    )

    para_at_peak = float(p_para[local_idx])
    transverse_to_compressive_at_peak = peak_power / para_at_peak if para_at_peak > 0 else np.inf
    para_left_floor = float(np.nanmin(finite_para[left_search])) if left_search.size else np.nan
    para_right_floor = float(np.nanmin(finite_para[right_search])) if right_search.size else np.nan
    para_side_floor = np.nanmax([para_left_floor, para_right_floor])
    if np.isfinite(para_side_floor) and para_side_floor > 0:
        para_side_prominence = para_at_peak / para_side_floor
    else:
        para_side_prominence = np.inf if para_at_peak > 0 else np.nan
    if np.isfinite(para_side_prominence) and para_side_prominence > 0:
        perp_para_peak_prominence_ratio = side_prominence / para_side_prominence
    else:
        perp_para_peak_prominence_ratio = np.inf if side_prominence > 0 else np.nan
    has_perp_peak_stronger_than_para = bool(
        np.isfinite(perp_para_peak_prominence_ratio)
        and perp_para_peak_prominence_ratio >= perp_para_peak_prominence_ratio_min
    )

    return {
        "has_peak": bool(has_two_sided_peak and prominence >= prominence_min),
        "frequency_hz": peak_frequency,
        "power": peak_power,
        "center_power": center_power,
        "low_neighbor_power": low_neighbor_power,
        "high_neighbor_power": high_neighbor_power,
        "background_power": float(background),
        "prominence": float(prominence),
        "left_floor_power": float(left_floor),
        "right_floor_power": float(right_floor),
        "side_floor_power": float(side_floor),
        "side_prominence": float(side_prominence),
        "center_median_power": float(center_median_power),
        "peak_to_band_median": float(peak_to_band_median),
        "half_prominence_level": float(half_prominence_level),
        "half_prominence_width_hz": float(half_prominence_width_hz),
        "half_prominence_width_factor": float(half_prominence_width_factor),
        "has_narrowband_peak": has_narrowband_peak,
        "adjacent_background_power": float(adjacent_background),
        "sharp_peak_adjacent_ratio": float(sharp_peak_adjacent_ratio),
        "has_sharp_peak": has_sharp_peak,
        "para_at_perp_peak_power": para_at_peak,
        "para_side_floor_power": float(para_side_floor),
        "para_side_prominence_at_perp_peak": float(para_side_prominence),
        "perp_para_peak_prominence_ratio": float(perp_para_peak_prominence_ratio),
        "has_perp_peak_stronger_than_para": has_perp_peak_stronger_than_para,
        "has_two_sided_peak": has_two_sided_peak,
        "is_local_max": is_local_max,
        "transverse_to_compressive_at_peak": float(transverse_to_compressive_at_peak),
    }


def detect_pickup_icw(
    Bwave: Any,
    ion: str | tuple[float, float] = "O+",
    *,
    criteria: ICWCriteria | None = None,
    psd_kwargs: dict[str, Any] | None = None,
    psd_result: Any | None = None,
    time_name: str | None = None,
    b_var: str | None = None,
    component_dim: str | None = None,
    nperseg: int | None = None,
    overlap: float = 0.5,
    reference_axis: np.ndarray | None = None,
    plot: bool = True,
    ax: Any = None,
    flip_ellipticity_sign: bool = False,
    min_gyro_bins: float = 4.0,
    detection_psd_source: str = "internal",
) -> dict[str, Any]:
    """
    Detect whether one xarray Bwave interval contains a pickup ICW.

    ``Bwave`` should be the same xarray-style magnetic-field object accepted by
    ``py_space_zc.method.psd_welch``. The function calls ``psd_welch`` to obtain
    perpendicular and parallel PSD, then applies the ICW thresholds from the
    literature. The polarization quantities are computed from the same magnetic
    interval using a Welch spectral matrix.
    """
    time, b_xyz_nt = _extract_bwave_arrays(
        Bwave,
        time_name=time_name,
        b_var=b_var,
        component_dim=component_dim,
    )
    return detect_pickup_icw_array(
        time,
        b_xyz_nt,
        ion=ion,
        criteria=criteria,
        psd_bwave=Bwave,
        psd_kwargs=psd_kwargs,
        psd_result=psd_result,
        nperseg=nperseg,
        overlap=overlap,
        reference_axis=reference_axis,
        plot=plot,
        ax=ax,
        flip_ellipticity_sign=flip_ellipticity_sign,
        min_gyro_bins=min_gyro_bins,
        detection_psd_source=detection_psd_source,
    )


def detect_pickup_icw_array(
    time: np.ndarray,
    b_xyz_nt: np.ndarray,
    ion: str | tuple[float, float] = "O+",
    *,
    criteria: ICWCriteria | None = None,
    psd_bwave: Any | None = None,
    psd_kwargs: dict[str, Any] | None = None,
    psd_result: Any | None = None,
    nperseg: int | None = None,
    overlap: float = 0.5,
    reference_axis: np.ndarray | None = None,
    plot: bool = True,
    ax: Any = None,
    flip_ellipticity_sign: bool = False,
    min_gyro_bins: float = 4.0,
    detection_psd_source: str = "internal",
) -> dict[str, Any]:
    """
    Detect whether one magnetic-field interval contains a pickup ICW.

    Parameters
    ----------
    time
        1D array of sample times. Numeric values are interpreted as seconds or
        any monotonic unit with seconds spacing. ``datetime64`` is also allowed.
    b_xyz_nt
        Nx3 magnetic field in nT, usually in MSO or MSE coordinates.
    ion
        Ion species, for example ``"O+"`` for pickup oxygen near Mars. A custom
        species can be passed as ``(mass_amu, charge_state)``.
    criteria
        Detection thresholds. Defaults follow common ICW event criteria.
    nperseg, overlap
        Welch parameters. For 20 Hz and 100 s windows, ``nperseg=512`` gives
        seven subintervals with 50 percent overlap, matching Schmid et al. style.
    reference_axis
        Reference axis used to define FAC/MFA perpendicular directions. Default
        is [1, 0, 0], matching ``py_space_zc.method.fac``.
    plot
        If True, draw time series in MFA coordinates and PSD diagnostics.
    ax
        Optional Matplotlib axes array with two or three axes.
    flip_ellipticity_sign
        Set True if your coordinate convention gives right-handed signs for
        known left-handed ICWs.
    min_gyro_bins
        When ``nperseg`` is not supplied, choose a Welch segment long enough
        that the frequency resolution is roughly ``fgyro / min_gyro_bins``.
        This matters for heavy pickup ions such as O+.

    Returns
    -------
    dict
        Detection result, summary metrics, frequency arrays, spectral products,
        MFA field, and the Matplotlib figure when ``plot=True``.
    """
    criteria = ICWCriteria() if criteria is None else criteria
    time = np.asarray(time)
    b_xyz_nt = np.asarray(b_xyz_nt, dtype=float)
    if b_xyz_nt.ndim != 2 or b_xyz_nt.shape[1] != 3:
        raise ValueError("b_xyz_nt must have shape (N, 3)")
    if time.shape[0] != b_xyz_nt.shape[0]:
        raise ValueError("time and b_xyz_nt must have the same length")

    good = np.all(np.isfinite(b_xyz_nt), axis=1) & np.isfinite(_as_seconds(time))
    if np.count_nonzero(good) < 8:
        raise ValueError("not enough finite samples")
    time_good = time[good]
    b_good = b_xyz_nt[good]

    fs = _sampling_rate_hz(time_good)
    b0 = np.nanmean(b_good, axis=0)
    b_sigma = float(np.nanstd(np.linalg.norm(b_good, axis=1)))
    basis = _mfa_basis(b0, reference_axis=reference_axis)
    b_mfa = (b_good - b0) @ basis.T

    mass, charge = _ion_mass_charge(ion)
    b0_t = np.linalg.norm(b0) * 1e-9
    sigma_b_t = b_sigma * 1e-9
    fgyro = abs(charge) * b0_t / (2.0 * np.pi * mass)
    dfgyro = abs(charge) * sigma_b_t / (2.0 * np.pi * mass)

    if nperseg is None and fgyro > 0:
        target = int(np.ceil(min_gyro_bins * fs / fgyro))
        target = 2 ** int(np.ceil(np.log2(max(target, 16))))
        nperseg = min(b_mfa.shape[0], max(512, target))

    freqs, smat = _welch_spectral_matrix(b_mfa, fs=fs, nperseg=nperseg, overlap=overlap)
    pol = _polarization_parameters(smat)
    if flip_ellipticity_sign:
        pol["ellipticity"] = -pol["ellipticity"]
        pol["circular_polarization"] = -pol["circular_polarization"]

    fmin = max(0.0, criteria.gyro_band_low_factor * fgyro - dfgyro)
    fmax = criteria.gyro_band_high_factor * fgyro + dfgyro
    low_neighbor_min = max(0.0, criteria.neighbor_band_low_factor * fgyro - dfgyro)
    low_neighbor_max = max(0.0, criteria.gyro_band_low_factor * fgyro - dfgyro)
    high_neighbor_min = criteria.gyro_band_high_factor * fgyro + dfgyro
    high_neighbor_max = criteria.neighbor_band_high_factor * fgyro + dfgyro
    df_res = float(freqs[1] - freqs[0]) if freqs.size > 1 else np.inf
    band = (freqs >= max(0.0, fmin - 0.5 * df_res)) & (freqs <= fmax + 0.5 * df_res)
    if not np.any(band):
        nearest = int(np.nanargmin(np.abs(freqs - fgyro)))
        band[nearest] = True

    psd_source = "internal_spectral_matrix"
    psd_welch_output = psd_result
    p_perp = pol["p_perp"]
    p_par = pol["p_par"]
    if psd_bwave is not None or psd_result is not None:
        if psd_welch_output is None:
            psd_welch_output = _call_psd_welch(psd_bwave, psd_kwargs)
        psd_freqs, p_perp_psd, p_par_psd = _extract_psd_welch_products(psd_welch_output)
        psd_freqs = np.asarray(psd_freqs, dtype=float)
        p_perp_psd = np.asarray(p_perp_psd, dtype=float).squeeze()
        p_par_psd = np.asarray(p_par_psd, dtype=float).squeeze()
        if p_perp_psd.shape != psd_freqs.shape or p_par_psd.shape != psd_freqs.shape:
            raise ValueError("psd_welch perp/para PSD must be 1D arrays matching the frequency coordinate")
        psd_source = "py_space_zc.method.psd_welch"
    else:
        psd_freqs = freqs
        p_perp_psd = p_perp
        p_par_psd = p_par

    if detection_psd_source not in {"internal", "psd_welch"}:
        raise ValueError('detection_psd_source must be "internal" or "psd_welch"')
    if detection_psd_source == "internal":
        det_freqs = freqs
        p_perp_det = p_perp
        p_par_det = p_par
    else:
        det_freqs = psd_freqs
        p_perp_det = p_perp_psd
        p_par_det = p_par_psd

    df_psd_res = float(psd_freqs[1] - psd_freqs[0]) if psd_freqs.size > 1 else np.inf
    psd_band = (psd_freqs >= max(0.0, fmin - 0.5 * df_psd_res)) & (psd_freqs <= fmax + 0.5 * df_psd_res)
    if not np.any(psd_band):
        nearest = int(np.nanargmin(np.abs(psd_freqs - fgyro)))
        psd_band[nearest] = True
    psd_low_neighbor_band = (
        (psd_freqs >= max(0.0, low_neighbor_min - 0.5 * df_psd_res))
        & (psd_freqs <= low_neighbor_max + 0.5 * df_psd_res)
    )
    psd_high_neighbor_band = (
        (psd_freqs >= max(0.0, high_neighbor_min - 0.5 * df_psd_res))
        & (psd_freqs <= high_neighbor_max + 0.5 * df_psd_res)
    )
    df_det_res = float(det_freqs[1] - det_freqs[0]) if det_freqs.size > 1 else np.inf
    det_band = (det_freqs >= max(0.0, fmin - 0.5 * df_det_res)) & (det_freqs <= fmax + 0.5 * df_det_res)
    if not np.any(det_band):
        nearest = int(np.nanargmin(np.abs(det_freqs - fgyro)))
        det_band[nearest] = True
    det_low_neighbor_band = (det_freqs >= low_neighbor_min) & (det_freqs < low_neighbor_max)
    det_high_neighbor_band = (det_freqs > high_neighbor_min) & (det_freqs <= high_neighbor_max)
    det_low_neighbor_band &= ~det_band
    det_high_neighbor_band &= ~det_band

    e_perp = _integrate_band(p_perp_det, det_freqs, det_band)
    e_par = _integrate_band(p_par_det, det_freqs, det_band)
    transverse_ratio = e_perp / e_par if e_par > 0 else np.inf

    weights = np.maximum(p_perp[det_band], 0.0)
    if np.sum(weights) > 0:
        ellipticity_band = float(np.average(pol["ellipticity"][det_band], weights=weights))
        dop_band = float(np.average(pol["dop"][det_band], weights=weights))
        wave_angle_band = float(np.nanmedian(pol["wave_angle_deg"][det_band]))
    else:
        ellipticity_band = float(np.nanmean(pol["ellipticity"][det_band]))
        dop_band = float(np.nanmean(pol["dop"][det_band]))
        wave_angle_band = float(np.nanmedian(pol["wave_angle_deg"][det_band]))

    p_total_psd = p_perp_psd + p_par_psd
    p_total_det = p_perp_det + p_par_det
    peak_metrics = _pcw_neighbor_peak_metrics(
        det_freqs,
        p_perp_det,
        p_par_det,
        det_band,
        det_low_neighbor_band,
        det_high_neighbor_band,
        criteria.peak_neighbor_prominence_min,
        criteria.narrowband_max_width_factor,
        criteria.narrowband_min_peak_to_band_median,
        criteria.perp_para_peak_prominence_ratio_min,
        criteria.sharp_peak_adjacent_ratio_min,
    )
    peak_frequency = peak_metrics["frequency_hz"]
    peak_in_band = peak_metrics["has_peak"]

    checks = {
        "perp_psd_peak_near_gyrofrequency": bool(peak_in_band or not criteria.require_peak_in_band),
        "perp_psd_peak_above_both_sides": bool(
            peak_metrics["has_two_sided_peak"] or not criteria.require_peak_above_both_sides
        ),
        "narrowband_peak": bool(peak_metrics["has_narrowband_peak"] or not criteria.require_narrowband_peak),
        "perp_peak_stronger_than_para": bool(
            peak_metrics["has_perp_peak_stronger_than_para"]
            or not criteria.require_perp_peak_stronger_than_para
        ),
        "sharp_perp_peak": bool(peak_metrics["has_sharp_peak"] or not criteria.require_sharp_perp_peak),
        "transverse_dominant": bool(transverse_ratio >= criteria.transverse_ratio_min),
        "left_hand_elliptical": bool(ellipticity_band <= criteria.ellipticity_max),
        "coherent": bool(dop_band >= criteria.dop_min),
        "quasi_parallel": bool(wave_angle_band <= criteria.wave_angle_max_deg),
    }
    is_icw = all(checks.values())
    duration_s = float(_as_seconds(time_good)[-1] - _as_seconds(time_good)[0])
    warnings = []
    if fgyro > 0 and duration_s * fgyro < 3.0:
        warnings.append(
            "The interval contains fewer than 3 target-ion gyroperiods; "
            "O+ ICW identification is likely under-resolved."
        )
    if freqs.size > 1 and (freqs[1] - freqs[0]) > fgyro / max(min_gyro_bins, 1.0):
        warnings.append(
            "The Welch frequency resolution is coarse relative to the target gyrofrequency; "
            "use a longer interval or smaller cadence after anti-alias filtering."
        )

    result: dict[str, Any] = {
        "is_icw": is_icw,
        "checks": checks,
        "ion": ion,
        "fs_hz": fs,
        "b0_nt": b0,
        "b0_abs_nt": float(np.linalg.norm(b0)),
        "b_sigma_nt": b_sigma,
        "fgyro_hz": float(fgyro),
        "dfgyro_hz": float(dfgyro),
        "frequency_band_hz": (float(fmin), float(fmax)),
        "low_neighbor_frequency_band_hz": (float(low_neighbor_min), float(low_neighbor_max)),
        "high_neighbor_frequency_band_hz": (float(high_neighbor_min), float(high_neighbor_max)),
        "frequency_resolution_hz": df_res,
        "transverse_ratio": float(transverse_ratio),
        "ellipticity": ellipticity_band,
        "degree_of_polarization": dop_band,
        "wave_angle_deg": wave_angle_band,
        "peak_frequency_hz": peak_frequency,
        "perp_psd_peak_frequency_hz": peak_metrics["frequency_hz"],
        "perp_psd_peak_power": peak_metrics["power"],
        "perp_psd_center_band_power": peak_metrics["center_power"],
        "perp_psd_low_neighbor_power": peak_metrics["low_neighbor_power"],
        "perp_psd_high_neighbor_power": peak_metrics["high_neighbor_power"],
        "perp_psd_peak_background_power": peak_metrics["background_power"],
        "perp_psd_peak_prominence": peak_metrics["prominence"],
        "perp_psd_peak_left_floor_power": peak_metrics["left_floor_power"],
        "perp_psd_peak_right_floor_power": peak_metrics["right_floor_power"],
        "perp_psd_peak_side_prominence": peak_metrics["side_prominence"],
        "perp_psd_peak_to_band_median": peak_metrics["peak_to_band_median"],
        "perp_psd_peak_half_prominence_width_hz": peak_metrics["half_prominence_width_hz"],
        "perp_psd_peak_half_prominence_width_factor": peak_metrics["half_prominence_width_factor"],
        "perp_psd_peak_has_narrowband_peak": peak_metrics["has_narrowband_peak"],
        "perp_psd_peak_adjacent_background_power": peak_metrics["adjacent_background_power"],
        "perp_psd_peak_sharp_adjacent_ratio": peak_metrics["sharp_peak_adjacent_ratio"],
        "perp_psd_peak_has_sharp_peak": peak_metrics["has_sharp_peak"],
        "para_psd_side_prominence_at_perp_peak": peak_metrics["para_side_prominence_at_perp_peak"],
        "perp_para_peak_prominence_ratio": peak_metrics["perp_para_peak_prominence_ratio"],
        "perp_peak_stronger_than_para": peak_metrics["has_perp_peak_stronger_than_para"],
        "perp_psd_peak_has_two_sided_peak": peak_metrics["has_two_sided_peak"],
        "perp_psd_peak_is_local_max": peak_metrics["is_local_max"],
        "perp_para_at_peak": peak_metrics["transverse_to_compressive_at_peak"],
        "total_psd_peak_frequency_hz": peak_metrics["frequency_hz"],
        "total_psd_peak_prominence": peak_metrics["prominence"],
        "psd_source": psd_source,
        "detection_psd_source": detection_psd_source,
        "psd_welch_output": psd_welch_output,
        "duration_s": duration_s,
        "nperseg": int(nperseg),
        "warnings": warnings,
        "freqs_hz": freqs,
        "psd_freqs_hz": psd_freqs,
        "detection_freqs_hz": det_freqs,
        "p_perp": p_perp,
        "p_par": p_par,
        "p_perp_psd_welch": p_perp_psd,
        "p_par_psd_welch": p_par_psd,
        "p_total_psd_welch": p_total_psd,
        "p_perp_detection": p_perp_det,
        "p_par_detection": p_par_det,
        "p_total_detection": p_total_det,
        "ellipticity_spectrum": pol["ellipticity"],
        "dop_spectrum": pol["dop"],
        "wave_angle_spectrum_deg": pol["wave_angle_deg"],
        "planarity_spectrum": pol["planarity"],
        "wave_normal_fac": pol["wave_normal_fac"],
        "b_mfa_nt": b_mfa,
        "mfa_basis_rows": basis,
        "spectral_matrix": smat,
    }

    if plot:
        result["figure"] = plot_icw_detection(time_good, b_mfa, result, ax=ax)

    return result


def plot_icw_detection(time: np.ndarray, b_mfa_nt: np.ndarray, result: dict[str, Any], ax: Any = None) -> Any:
    """Plot MFA magnetic field, PSD, and polarization diagnostics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot=True requires matplotlib") from exc

    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), constrained_layout=True)
    else:
        axes = np.ravel(ax)
        fig = axes[0].figure
        if axes.size < 3:
            raise ValueError("ax must contain at least three axes")

    tsec = _as_seconds(time)
    freqs = result.get("psd_freqs_hz", result["freqs_hz"])
    p_perp = result.get("p_perp_psd_welch", result["p_perp"])
    p_par = result.get("p_par_psd_welch", result["p_par"])
    fmin, fmax = result["frequency_band_hz"]
    stride = max(1, int(np.ceil(tsec.size / 6000)))
    t_plot = tsec[::stride]
    b_plot = b_mfa_nt[::stride]

    axes[0].plot(t_plot, b_plot[:, 0], color="tab:red", lw=1.2, label="Bperp1")
    axes[0].plot(t_plot, b_plot[:, 1], color="tab:green", lw=1.2, label="Bperp2")
    axes[0].plot(t_plot, b_plot[:, 2], color="tab:blue", lw=1.2, label="Bparallel")
    axes[0].set_ylabel("dB MFA (nT)")
    axes[0].grid(True, ls=":", alpha=0.35)
    axes[0].legend(loc="upper right", ncol=3, fontsize=9, frameon=True)

    axes[1].loglog(freqs[1:], p_perp[1:], color="tab:red", lw=1.8, label="Pperp")
    axes[1].loglog(freqs[1:], p_par[1:], color="tab:blue", lw=1.8, label="Pparallel")
    axes[1].axvline(result["fgyro_hz"], color="tab:green", lw=1.4, label="fgyro")
    axes[1].axvspan(fmin, fmax, color="0.88", zorder=0, label="test band")
    axes[1].axvline(result["peak_frequency_hz"], color="tab:red", lw=1.0, ls=":", label="Pperp peak")
    axes[1].set_ylabel("PSD (nT^2/Hz)")
    axes[1].grid(True, which="both", ls=":", alpha=0.35)
    axes[1].legend(loc="best", fontsize=9, frameon=True)

    pol_freqs = result["freqs_hz"]
    axes[2].plot(pol_freqs, result["ellipticity_spectrum"], color="tab:red", lw=1.5, label="ellipticity")
    axes[2].plot(pol_freqs, result["dop_spectrum"], color="tab:blue", lw=1.5, label="DOP")
    axes[2].axhline(-0.55, color="tab:red", ls=":", lw=1)
    axes[2].axhline(0.7, color="tab:green", ls=":", lw=1)
    axes[2].axvline(result["fgyro_hz"], color="tab:green", lw=1.4)
    axes[2].axvspan(fmin, fmax, color="0.88", zorder=0)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Polarization")
    axes[2].grid(True, ls=":", alpha=0.35)
    axes[2].legend(loc="best", fontsize=9, frameon=True)

    title = "ICW" if result["is_icw"] else "not ICW"
    axes[0].set_title(
        f"{title}: {result['ion']}, fgyro={result['fgyro_hz']:.4g} Hz, "
        f"Eperp/Epar={result['transverse_ratio']:.2f}, eps={result['ellipticity']:.2f}, "
        f"Pperp peak/Ppar peak prom={result.get('perp_para_peak_prominence_ratio', np.nan):.2f}"
    )
    return fig


def _time64_to_datetime64(value: Any) -> np.datetime64:
    text = str(value).strip().replace("/", "T")
    if text.endswith(".0"):
        text = text[:-2]
    return np.datetime64(text)


def _format_datetime64(value: Any) -> str:
    return np.datetime_as_string(np.asarray(value).astype("datetime64[s]"), unit="s")


def _datetime64_to_seconds(time: np.ndarray) -> np.ndarray:
    time = np.asarray(time)
    if time.size == 0:
        return np.asarray([], dtype=float)
    if np.issubdtype(time.dtype, np.datetime64):
        return (time - time[0]) / np.timedelta64(1, "s")
    return time.astype(float) - float(time[0])


def read_sw_interval_list(path: str, min_duration_s: float = 40.0 * 60.0) -> list[dict[str, Any]]:
    """
    Read the CNN solar-wind interval list.

    Expected rows look like:
    ``0 2014-12-01/01:05:30.0 2014-12-01/03:17:00.0 2.2``
    """
    intervals: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                index = int(parts[0])
                start = _time64_to_datetime64(parts[1])
                stop = _time64_to_datetime64(parts[2])
            except (TypeError, ValueError):
                continue
            duration_s = float((stop - start) / np.timedelta64(1, "s"))
            if duration_s >= min_duration_s:
                intervals.append(
                    {
                        "index": index,
                        "start": start,
                        "stop": stop,
                        "duration_s": duration_s,
                        "line_no": line_no,
                    }
                )
    return intervals


def integrate_frequency_band(freq: np.ndarray, psd: np.ndarray, f_band: tuple[float, float]) -> float:
    freq = np.asarray(freq, dtype=float)
    psd = np.asarray(psd, dtype=float)
    band = (freq >= f_band[0]) & (freq <= f_band[1])
    if not np.any(band):
        band[int(np.nanargmin(np.abs(freq - np.nanmean(f_band))))] = True
    if np.count_nonzero(band) == 1:
        return float(psd[band][0])
    return float(np.trapz(psd[band], freq[band]))


def sliding_pickup_icw_detection(
    Bwave: Any,
    ion: str | tuple[float, float] = "O+",
    *,
    window_length_s: float = 40.0 * 60.0,
    overlap: float = 0.5,
    criteria: ICWCriteria | None = None,
    detection_psd_source: str = "internal",
    min_samples: int = 64,
    **detect_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Run pickup ICW detection in sliding windows.

    For O+ at a typical 3 nT IMF, 40 minutes contains about 6 gyroperiods,
    so this is the default window length.
    """
    time, _ = _extract_bwave_arrays(Bwave)
    time_sec = _datetime64_to_seconds(time)
    if time_sec.size < min_samples:
        return []

    duration_s = float(time_sec[-1] - time_sec[0])
    if duration_s < window_length_s:
        return []

    step_s = window_length_s * (1.0 - overlap)
    if step_s <= 0:
        raise ValueError("overlap must be smaller than 1")

    rows: list[dict[str, Any]] = []
    start_s = 0.0
    while start_s + window_length_s <= duration_s + 1e-6:
        stop_s = start_s + window_length_s
        idx = (time_sec >= start_s) & (time_sec < stop_s)
        if np.count_nonzero(idx) < min_samples:
            start_s += step_s
            continue

        Bwin = Bwave.isel(time=idx)
        result = detect_pickup_icw(
            Bwin,
            ion=ion,
            criteria=criteria,
            plot=False,
            detection_psd_source=detection_psd_source,
            **detect_kwargs,
        )
        freq = result["detection_freqs_hz"]
        para_psd = result["p_par_detection"]
        perp_psd = result["p_perp_detection"]
        f_band = result["frequency_band_hz"]
        e_para = integrate_frequency_band(freq, para_psd, f_band)
        e_perp = integrate_frequency_band(freq, perp_psd, f_band)

        rows.append(
            {
                "start": np.asarray(Bwin.time.data[0]).astype("datetime64[s]"),
                "stop": np.asarray(Bwin.time.data[-1]).astype("datetime64[s]"),
                "center": np.asarray(Bwin.time.data[len(Bwin.time) // 2]).astype("datetime64[s]"),
                "is_icw": bool(result["is_icw"]),
                "ion": result["ion"],
                "b0_abs_nt": result["b0_abs_nt"],
                "fgyro_hz": result["fgyro_hz"],
                "gyroperiod_s": 1.0 / result["fgyro_hz"] if result["fgyro_hz"] > 0 else np.nan,
                "frequency_band_hz": result["frequency_band_hz"],
                "peak_frequency_hz": result["peak_frequency_hz"],
                "perp_para_ratio": result["transverse_ratio"],
                "ellipticity": result["ellipticity"],
                "degree_of_polarization": result["degree_of_polarization"],
                "wave_angle_deg": result["wave_angle_deg"],
                "peak_prominence": result["perp_psd_peak_prominence"],
                "peak_side_prominence": result["perp_psd_peak_side_prominence"],
                "peak_sharp_adjacent_ratio": result["perp_psd_peak_sharp_adjacent_ratio"],
                "peak_width_factor": result["perp_psd_peak_half_prominence_width_factor"],
                "peak_to_band_median": result["perp_psd_peak_to_band_median"],
                "para_peak_side_prominence": result["para_psd_side_prominence_at_perp_peak"],
                "perp_para_peak_prominence_ratio": result["perp_para_peak_prominence_ratio"],
                "e_para": e_para,
                "e_perp": e_perp,
                "checks": result["checks"],
                "warnings": "; ".join(result["warnings"]),
            }
        )
        start_s += step_s

    return rows


def save_icw_rows(rows: list[dict[str, Any]], csv_path: str, true_path: str | None = None) -> None:
    import csv

    fieldnames = [
        "source_index",
        "start",
        "stop",
        "center",
        "is_icw",
        "ion",
        "b0_abs_nt",
        "fgyro_hz",
        "gyroperiod_s",
        "peak_frequency_hz",
        "perp_para_ratio",
        "ellipticity",
        "degree_of_polarization",
        "wave_angle_deg",
        "peak_prominence",
        "peak_side_prominence",
        "peak_sharp_adjacent_ratio",
        "peak_width_factor",
        "peak_to_band_median",
        "para_peak_side_prominence",
        "perp_para_peak_prominence_ratio",
        "e_para",
        "e_perp",
        "checks",
        "warnings",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("start", "stop", "center"):
                out[key] = _format_datetime64(out[key])
            out["checks"] = "; ".join(f"{k}={v}" for k, v in out.get("checks", {}).items())
            writer.writerow(out)

    if true_path is not None:
        with open(true_path, "w", encoding="utf-8") as f:
            for row in rows:
                if row.get("is_icw"):
                    f.write(f"{_format_datetime64(row['start'])} {_format_datetime64(row['stop'])}\n")


def plot_icw_window_diagnostic(
    Bwave: Any,
    row: dict[str, Any],
    path: str,
    *,
    ion: str | tuple[float, float] = "O+",
    criteria: ICWCriteria | None = None,
    detection_psd_source: str = "internal",
    dpi: int = 180,
    **detect_kwargs: Any,
) -> None:
    import matplotlib.pyplot as plt

    time, _ = _extract_bwave_arrays(Bwave)
    t0 = np.asarray(row["start"]).astype("datetime64[ns]")
    t1 = np.asarray(row["stop"]).astype("datetime64[ns]")
    idx = (time >= t0) & (time <= t1)
    Bwin = Bwave.isel(time=idx)
    result = detect_pickup_icw(
        Bwin,
        ion=ion,
        criteria=criteria,
        plot=True,
        detection_psd_source=detection_psd_source,
        **detect_kwargs,
    )
    fig = result["figure"]
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _longest_true_run_duration(time: np.ndarray, mask: np.ndarray) -> float:
    time = np.asarray(time)
    mask = np.asarray(mask, dtype=bool)
    if time.size == 0 or not np.any(mask):
        return 0.0
    if time.size == 1:
        return 0.0
    tsec = _datetime64_to_seconds(time)
    dt = float(np.nanmedian(np.diff(tsec))) if tsec.size > 1 else 0.0
    best = 0.0
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < mask.size and mask[j + 1]:
            j += 1
        best = max(best, float(tsec[j] - tsec[i] + dt))
        i = j + 1
    return best


def _svd_frequency_prominence(freq: np.ndarray, power_tf: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    prominence = np.full(power_tf.shape[0], np.nan)
    for i, idx in enumerate(peak_indices):
        if idx < 0 or idx >= freq.size:
            continue
        peak = power_tf[i, idx]
        if not np.isfinite(peak) or peak <= 0:
            continue
        left = max(0, idx - 3)
        right = min(freq.size, idx + 4)
        bg_idx = [j for j in range(left, right) if abs(j - idx) >= 1]
        if not bg_idx:
            continue
        bg = np.nanmedian(power_tf[i, bg_idx])
        prominence[i] = peak / bg if np.isfinite(bg) and bg > 0 else np.inf
    return prominence


def detect_pcw_svd(
    Bwave: Any,
    ion: str | tuple[float, float] = "H+",
    *,
    criteria: PCWSVDCriteria | None = None,
    window_length: float = 80.0,
    overlap: float = 40.0,
    freq_range: tuple[float, float] | None = None,
    tint_focus: tuple[str, str] | list[str] | tuple[np.datetime64, np.datetime64] | None = None,
    plot: bool = False,
    m_width_coeff: float = 1,
) -> dict[str, Any]:
    """
    Detect proton cyclotron wave intervals using SVD/wavelet time-frequency products.

    The test is designed for 1 s MAVEN MAG data and PCW periods of about
    10 to 50 s. It uses ``py_space_zc.method.SVD_B`` to obtain perpendicular
    and parallel PSD, wave-normal angle, ellipticity, and planarity.
    """
    criteria = PCWSVDCriteria() if criteria is None else criteria
    time, b_xyz_nt = _extract_bwave_arrays(Bwave)
    good = np.all(np.isfinite(b_xyz_nt), axis=1)
    if np.count_nonzero(good) < 32:
        raise ValueError("not enough finite samples for SVD PCW detection")

    mass, charge = _ion_mass_charge(ion)
    b0_abs_nt = float(np.linalg.norm(np.nanmean(b_xyz_nt[good], axis=0)))
    fgyro_hz = abs(charge) * b0_abs_nt * 1e-9 / (2.0 * np.pi * mass)
    pcw_band = (1.0 / criteria.period_range_s[1], 1.0 / criteria.period_range_s[0])
    gyro_band = (
        criteria.gyro_band_low_factor * fgyro_hz,
        criteria.gyro_band_high_factor * fgyro_hz,
    )
    band_low = max(pcw_band[0], gyro_band[0])
    band_high = min(pcw_band[1], gyro_band[1])
    if band_low >= band_high:
        band_low, band_high = pcw_band
    if freq_range is None:
        freq_range = (max(0.005, 0.5 * band_low), min(0.5, 2.0 * band_high))

    try:
        from py_space_zc import method
    except ImportError as exc:
        raise ImportError("py_space_zc is not available, cannot call method.SVD_B") from exc

    wave_res = method.SVD_B(
        Bwave.dropna(dim="time") if hasattr(Bwave, "dropna") else Bwave,
        window_length=window_length,
        overlap=overlap,
        freq_range=list(freq_range),
        m_width_coeff=m_width_coeff,
    )
    bperp = np.asarray(wave_res["Bperp"].values, dtype=float)
    bpara = np.asarray(wave_res["Bpara"].values, dtype=float)
    ellipticity = np.asarray(wave_res["ellipticity"].values, dtype=float)
    theta = np.asarray(wave_res["theta"].values, dtype=float)
    planarity = np.asarray(wave_res["planarity"].values, dtype=float)
    svd_time = np.asarray(wave_res["Bperp"].time.data)
    freq = np.asarray(wave_res["Bperp"].coords["frequency"].data, dtype=float)
    if tint_focus is None:
        focus_mask = np.ones(svd_time.size, dtype=bool)
        focus_tint = (svd_time[0], svd_time[-1]) if svd_time.size else (None, None)
    else:
        focus_tint = (np.datetime64(tint_focus[0]), np.datetime64(tint_focus[1]))
        focus_mask = (svd_time >= focus_tint[0]) & (svd_time <= focus_tint[1])
        if not np.any(focus_mask):
            raise ValueError("tint_focus does not overlap the SVD time axis")

    band = (freq >= band_low) & (freq <= band_high)
    if not np.any(band):
        nearest = int(np.nanargmin(np.abs(freq - fgyro_hz)))
        band[nearest] = True
    band_indices = np.flatnonzero(band)
    peak_indices = np.full(svd_time.size, -1, dtype=int)
    for i in range(svd_time.size):
        if np.any(np.isfinite(bperp[i, band_indices])):
            peak_indices[i] = band_indices[int(np.nanargmax(bperp[i, band_indices]))]

    row = np.arange(svd_time.size)
    valid_peak = peak_indices >= 0
    peak_power = np.full(svd_time.size, np.nan)
    para_peak_power = np.full(svd_time.size, np.nan)
    peak_freq = np.full(svd_time.size, np.nan)
    peak_ellipticity = np.full(svd_time.size, np.nan)
    peak_theta = np.full(svd_time.size, np.nan)
    peak_planarity = np.full(svd_time.size, np.nan)
    peak_power[valid_peak] = bperp[row[valid_peak], peak_indices[valid_peak]]
    para_peak_power[valid_peak] = bpara[row[valid_peak], peak_indices[valid_peak]]
    peak_freq[valid_peak] = freq[peak_indices[valid_peak]]
    peak_ellipticity[valid_peak] = ellipticity[row[valid_peak], peak_indices[valid_peak]]
    peak_theta[valid_peak] = theta[row[valid_peak], peak_indices[valid_peak]]
    peak_planarity[valid_peak] = planarity[row[valid_peak], peak_indices[valid_peak]]

    with np.errstate(divide="ignore", invalid="ignore"):
        transverse_ratio = peak_power / para_peak_power
    perp_prominence = _svd_frequency_prominence(freq, bperp, peak_indices)
    para_prominence = _svd_frequency_prominence(freq, bpara, peak_indices)
    with np.errstate(divide="ignore", invalid="ignore"):
        prominence_ratio = perp_prominence / para_prominence

    good_bins = (
        valid_peak
        & (transverse_ratio >= criteria.transverse_ratio_min)
        & (peak_ellipticity <= criteria.ellipticity_max)
        & (peak_theta <= criteria.wave_angle_max_deg)
        & (peak_planarity >= criteria.planarity_min)
        & (perp_prominence >= criteria.narrowband_prominence_min)
        & (prominence_ratio >= criteria.perp_para_peak_prominence_ratio_min)
    )
    good_focus_bins = good_bins & focus_mask
    focus_count = int(np.count_nonzero(focus_mask))
    good_fraction = float(np.count_nonzero(good_focus_bins) / max(focus_count, 1))
    longest_duration_s = _longest_true_run_duration(svd_time[focus_mask], good_bins[focus_mask])
    median_good_prominence = (
        float(np.nanmedian(perp_prominence[good_focus_bins])) if np.any(good_focus_bins) else np.nan
    )
    checks = {
        "enough_good_windows": bool(np.count_nonzero(good_focus_bins) >= criteria.min_good_windows),
        "enough_good_time_fraction": bool(good_fraction >= criteria.min_good_time_fraction),
        "continuous_enough": bool(longest_duration_s >= criteria.min_continuous_duration_s),
        "strong_median_narrowband_prominence": bool(
            np.isfinite(median_good_prominence)
            and median_good_prominence >= criteria.median_narrowband_prominence_min
        ),
    }
    is_pcw = all(checks.values())

    result: dict[str, Any] = {
        "is_icw": is_pcw,
        "is_pcw": is_pcw,
        "checks": checks,
        "criteria": criteria,
        "ion": ion,
        "b0_abs_nt": b0_abs_nt,
        "fgyro_hz": float(fgyro_hz),
        "pcw_frequency_band_hz": (float(pcw_band[0]), float(pcw_band[1])),
        "gyro_frequency_band_hz": (float(gyro_band[0]), float(gyro_band[1])),
        "detection_band_hz": (float(band_low), float(band_high)),
        "freqs_hz": freq,
        "time": svd_time,
        "focus_tint": focus_tint,
        "focus_time_mask": focus_mask,
        "Bperp": bperp,
        "Bpara": bpara,
        "ellipticity": ellipticity,
        "theta_deg": theta,
        "planarity": planarity,
        "peak_frequency_hz": peak_freq,
        "peak_transverse_ratio": transverse_ratio,
        "peak_ellipticity": peak_ellipticity,
        "peak_theta_deg": peak_theta,
        "peak_planarity": peak_planarity,
        "perp_peak_prominence": perp_prominence,
        "para_peak_prominence": para_prominence,
        "perp_para_prominence_ratio": prominence_ratio,
        "good_time_mask": good_bins,
        "good_focus_time_mask": good_focus_bins,
        "good_time_fraction": good_fraction,
        "good_window_count": int(np.count_nonzero(good_focus_bins)),
        "focus_window_count": focus_count,
        "longest_good_duration_s": float(longest_duration_s),
        "median_good_perp_prominence": median_good_prominence,
        "wave_res": wave_res,
    }
    if plot:
        result["figure"] = plot_pcw_svd_detection(Bwave, result)
    return result


def _positive_norm(data: np.ndarray) -> Any:
    try:
        from matplotlib.colors import LogNorm
    except ImportError:
        return None
    finite = np.asarray(data)[np.isfinite(data) & (np.asarray(data) > 0)]
    if finite.size == 0:
        return None
    return LogNorm(vmin=np.nanpercentile(finite, 5), vmax=np.nanpercentile(finite, 98))


def plot_pcw_svd_detection(Bwave: Any, result: dict[str, Any]) -> Any:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    time = result["time"].astype("datetime64[ms]").astype(object)
    t_num = mdates.date2num(time)
    freq = result["freqs_hz"]
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True, constrained_layout=True)

    Btime, Bdata = _extract_bwave_arrays(Bwave)
    btime = Btime.astype("datetime64[ms]").astype(object)
    axes[0].plot(btime, Bdata[:, 0], color="tab:red", lw=1.0, label="Bx")
    axes[0].plot(btime, Bdata[:, 1], color="tab:green", lw=1.0, label="By")
    axes[0].plot(btime, Bdata[:, 2], color="tab:blue", lw=1.0, label="Bz")
    axes[0].set_ylabel("B MSO\n(nT)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].grid(True, ls=":", alpha=0.35)

    panels = [
        ("Bperp PSD", result["Bperp"], "Spectral_r", _positive_norm(result["Bperp"])),
        ("Bpara PSD", result["Bpara"], "Spectral_r", _positive_norm(result["Bpara"])),
        ("Bperp/Bpara", result["Bperp"] / result["Bpara"], "coolwarm", None),
        ("Theta kB (deg)", result["theta_deg"], "coolwarm", None),
        ("Ellipticity", result["ellipticity"], "coolwarm", None),
    ]
    for ax, (label, data, cmap, norm) in zip(axes[1:], panels):
        pc = ax.pcolormesh(t_num, freq, np.asarray(data).T, shading="auto", cmap=cmap, norm=norm)
        ax.axhline(result["fgyro_hz"], color="k", lw=1.2, label="fcp")
        ax.axhspan(*result["detection_band_hz"], color="0.8", alpha=0.5)
        ax.set_yscale("log")
        ax.set_ylabel("Freq\n(Hz)")
        ax.set_ylim(max(freq[0], 0.008), freq[-1])
        ax.grid(True, which="both", ls=":", alpha=0.25)
        ax.text(0.01, 0.88, label, transform=ax.transAxes, color="black", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        fig.colorbar(pc, ax=ax, pad=0.01)

    focus_time = time[result["focus_time_mask"]]
    good_time = time[result["good_focus_time_mask"]]
    for ax in axes[1:]:
        if focus_time.size:
            ax.axvspan(focus_time[0], focus_time[-1], color="0.9", alpha=0.45, zorder=0)
        for t in good_time:
            ax.axvline(t, color="tab:red", lw=0.7, alpha=0.35)

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    title = "PCW" if result["is_pcw"] else "not PCW"
    axes[0].set_title(
        f"{title}: {result['ion']}, fcp={result['fgyro_hz']:.4f} Hz, "
        f"good={result['good_time_fraction']:.2f}, duration={result['longest_good_duration_s']:.0f} s"
    )
    return fig


def _pcw_svd_metrics_for_window(
    freq: np.ndarray,
    bperp: np.ndarray,
    bpara: np.ndarray,
    ellipticity: np.ndarray,
    theta: np.ndarray,
    planarity: np.ndarray,
    band: np.ndarray,
    criteria: PCWSVDCriteria,
) -> dict[str, Any]:
    if bperp.size == 0 or not np.any(band):
        return {
            "is_pcw": False,
            "good_fraction": 0.0,
            "good_count": 0,
            "median_perp_para": np.nan,
            "median_theta_deg": np.nan,
            "median_ellipticity": np.nan,
            "median_planarity": np.nan,
            "median_perp_prominence": np.nan,
            "median_peak_frequency_hz": np.nan,
        }

    band_idx = np.flatnonzero(band)
    peak_idx = np.full(bperp.shape[0], -1, dtype=int)
    for i in range(bperp.shape[0]):
        if np.any(np.isfinite(bperp[i, band_idx])):
            peak_idx[i] = band_idx[int(np.nanargmax(bperp[i, band_idx]))]

    row = np.arange(bperp.shape[0])
    valid = peak_idx >= 0
    peak_perp = np.full(bperp.shape[0], np.nan)
    peak_para = np.full(bperp.shape[0], np.nan)
    peak_freq = np.full(bperp.shape[0], np.nan)
    peak_ell = np.full(bperp.shape[0], np.nan)
    peak_theta = np.full(bperp.shape[0], np.nan)
    peak_planarity = np.full(bperp.shape[0], np.nan)
    peak_perp[valid] = bperp[row[valid], peak_idx[valid]]
    peak_para[valid] = bpara[row[valid], peak_idx[valid]]
    peak_freq[valid] = freq[peak_idx[valid]]
    peak_ell[valid] = ellipticity[row[valid], peak_idx[valid]]
    peak_theta[valid] = theta[row[valid], peak_idx[valid]]
    peak_planarity[valid] = planarity[row[valid], peak_idx[valid]]

    with np.errstate(divide="ignore", invalid="ignore"):
        perp_para = peak_perp / peak_para
    perp_prom = _svd_frequency_prominence(freq, bperp, peak_idx)
    para_prom = _svd_frequency_prominence(freq, bpara, peak_idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        prom_ratio = perp_prom / para_prom

    good = (
        valid
        & (perp_para >= criteria.transverse_ratio_min)
        & (peak_ell <= criteria.ellipticity_max)
        & (peak_theta <= criteria.wave_angle_max_deg)
        & (peak_planarity >= criteria.planarity_min)
        & (perp_prom >= criteria.narrowband_prominence_min)
        & (prom_ratio >= criteria.perp_para_peak_prominence_ratio_min)
    )
    good_fraction = float(np.count_nonzero(good) / max(good.size, 1))
    median_perp_prom = float(np.nanmedian(perp_prom[good])) if np.any(good) else np.nan
    is_pcw = bool(
        np.count_nonzero(good) >= criteria.min_good_windows
        and good_fraction >= criteria.min_good_time_fraction
        and np.isfinite(median_perp_prom)
        and median_perp_prom >= criteria.median_narrowband_prominence_min
    )

    return {
        "is_pcw": is_pcw,
        "good_fraction": good_fraction,
        "good_count": int(np.count_nonzero(good)),
        "median_perp_para": float(np.nanmedian(perp_para[good])) if np.any(good) else np.nan,
        "median_theta_deg": float(np.nanmedian(peak_theta[good])) if np.any(good) else np.nan,
        "median_ellipticity": float(np.nanmedian(peak_ell[good])) if np.any(good) else np.nan,
        "median_planarity": float(np.nanmedian(peak_planarity[good])) if np.any(good) else np.nan,
        "median_perp_prominence": median_perp_prom,
        "median_peak_frequency_hz": float(np.nanmedian(peak_freq[good])) if np.any(good) else np.nan,
    }


def detect_pcw_svd_5min(
    Bwave: Any,
    ion: str | tuple[float, float] = "H+",
    *,
    criteria: PCWSVDCriteria | None = None,
    window_s: float = 300.0,
    step_s: float = 300.0,
    svd_window_length: float = 80.0,
    svd_overlap: float = 40.0,
    freq_range: tuple[float, float] = (0.01, 0.2),
    tint_focus: tuple[str, str] | list[str] | tuple[np.datetime64, np.datetime64] | None = None,
    plot: bool = False,
    m_width_coeff: float = 1,
) -> dict[str, Any]:
    """Detect PCW in independent 5 minute windows using local gyrofrequency."""
    criteria = PCWSVDCriteria() if criteria is None else criteria
    time, b_xyz_nt = _extract_bwave_arrays(Bwave)
    if tint_focus is None:
        focus_start = time[0]
        focus_stop = time[-1]
    else:
        focus_start = np.datetime64(tint_focus[0])
        focus_stop = np.datetime64(tint_focus[1])

    try:
        from py_space_zc import method
    except ImportError as exc:
        raise ImportError("py_space_zc is not available, cannot call method.SVD_B") from exc
    wave_res = method.SVD_B(
        Bwave.dropna(dim="time") if hasattr(Bwave, "dropna") else Bwave,
        window_length=svd_window_length,
        overlap=svd_overlap,
        freq_range=list(freq_range),
        m_width_coeff=m_width_coeff,
    )
    svd_time = np.asarray(wave_res["Bperp"].time.data)
    freq = np.asarray(wave_res["Bperp"].coords["frequency"].data, dtype=float)
    bperp = np.asarray(wave_res["Bperp"].values, dtype=float)
    bpara = np.asarray(wave_res["Bpara"].values, dtype=float)
    ellipticity = np.asarray(wave_res["ellipticity"].values, dtype=float)
    theta = np.asarray(wave_res["theta"].values, dtype=float)
    planarity = np.asarray(wave_res["planarity"].values, dtype=float)

    mass, charge = _ion_mass_charge(ion)
    pcw_band = (1.0 / criteria.period_range_s[1], 1.0 / criteria.period_range_s[0])
    rows: list[dict[str, Any]] = []
    start = np.datetime64(focus_start)
    step = np.timedelta64(int(round(step_s)), "s")
    width = np.timedelta64(int(round(window_s)), "s")
    while start + width <= np.datetime64(focus_stop) + np.timedelta64(1, "s"):
        stop = start + width
        bmask = (time >= start) & (time < stop)
        smask = (svd_time >= start) & (svd_time < stop)
        if np.count_nonzero(bmask) < 16 or np.count_nonzero(smask) == 0:
            start = start + step
            continue
        b0_abs_nt = float(np.linalg.norm(np.nanmean(b_xyz_nt[bmask], axis=0)))
        fgyro = abs(charge) * b0_abs_nt * 1e-9 / (2.0 * np.pi * mass)
        gyro_band = (
            criteria.gyro_band_low_factor * fgyro,
            criteria.gyro_band_high_factor * fgyro,
        )
        band_low = max(pcw_band[0], gyro_band[0])
        band_high = min(pcw_band[1], gyro_band[1])
        if band_low >= band_high:
            band_low, band_high = gyro_band
        band = (freq >= band_low) & (freq <= band_high)
        metrics = _pcw_svd_metrics_for_window(
            freq,
            bperp[smask],
            bpara[smask],
            ellipticity[smask],
            theta[smask],
            planarity[smask],
            band,
            criteria,
        )
        rows.append(
            {
                "start": start.astype("datetime64[s]"),
                "stop": stop.astype("datetime64[s]"),
                "center": (start + width // 2).astype("datetime64[s]"),
                "b0_abs_nt": b0_abs_nt,
                "fgyro_hz": float(fgyro),
                "band_low_hz": float(band_low),
                "band_high_hz": float(band_high),
                **metrics,
            }
        )
        start = start + step

    is_pcw = bool(any(row["is_pcw"] for row in rows))
    result: dict[str, Any] = {
        "is_pcw": is_pcw,
        "is_icw": is_pcw,
        "ion": ion,
        "criteria": criteria,
        "window_s": window_s,
        "step_s": step_s,
        "tint_focus": (focus_start, focus_stop),
        "rows": rows,
        "time": svd_time,
        "freqs_hz": freq,
        "Bperp": bperp,
        "Bpara": bpara,
        "ellipticity": ellipticity,
        "theta_deg": theta,
        "planarity": planarity,
        "wave_res": wave_res,
    }
    if plot:
        result["figure"] = plot_pcw_svd_5min_detection(Bwave, result)
    return result


def pcw_events_from_5min_rows(
    rows: list[dict[str, Any]],
    *,
    min_consecutive_points: int = 4,
    window_s: float = 180.0,
) -> list[dict[str, Any]]:
    """Group consecutive 5 min PCW-positive rows into events."""
    events: list[dict[str, Any]] = []
    i = 0
    while i < len(rows):
        if not rows[i].get("is_pcw", False):
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and rows[j + 1].get("is_pcw", False):
            j += 1
        run = rows[i : j + 1]
        if len(run) >= min_consecutive_points:
            events.append(
                {
                    "event_start": run[0]["start"],
                    "event_stop": run[-1]["stop"],
                    "window_count": len(run),
                    "effective_duration_s": float(len(run) * window_s),
                    "covered_duration_s": float(
                        (np.datetime64(run[-1]["stop"]) - np.datetime64(run[0]["start"]))
                        / np.timedelta64(1, "s")
                    ),
                    "median_fgyro_hz": float(np.nanmedian([x["fgyro_hz"] for x in run])),
                    "median_peak_frequency_hz": float(np.nanmedian([x["median_peak_frequency_hz"] for x in run])),
                    "median_perp_para": float(np.nanmedian([x["median_perp_para"] for x in run])),
                    "median_theta_deg": float(np.nanmedian([x["median_theta_deg"] for x in run])),
                    "median_ellipticity": float(np.nanmedian([x["median_ellipticity"] for x in run])),
                    "median_planarity": float(np.nanmedian([x["median_planarity"] for x in run])),
                    "median_perp_prominence": float(np.nanmedian([x["median_perp_prominence"] for x in run])),
                    "mean_good_fraction": float(np.nanmean([x["good_fraction"] for x in run])),
                }
            )
        i = j + 1
    return events


def plot_pcw_svd_5min_detection(Bwave: Any, result: dict[str, Any], Bplot: Any | None = None) -> Any:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    time = result["time"].astype("datetime64[ms]").astype(object)
    t_num = mdates.date2num(time)
    freq = result["freqs_hz"]
    rows = result["rows"]
    focus_start, focus_stop = result["tint_focus"]
    focus_xlim = [
        np.datetime64(focus_start).astype("datetime64[ms]").astype(object),
        np.datetime64(focus_stop).astype("datetime64[ms]").astype(object),
    ]
    fig, axes = plt.subplots(7, 1, figsize=(13, 12), sharex=True, constrained_layout=True)

    plot_source = Bwave if Bplot is None else Bplot
    btime, bdata = _extract_bwave_arrays(plot_source)
    focus_bmask = (btime >= np.datetime64(focus_start)) & (btime <= np.datetime64(focus_stop))
    axes[0].plot(btime.astype("datetime64[ms]").astype(object), bdata[:, 0], color="tab:red", lw=1.0, label="Bx")
    axes[0].plot(btime.astype("datetime64[ms]").astype(object), bdata[:, 1], color="tab:green", lw=1.0, label="By")
    axes[0].plot(btime.astype("datetime64[ms]").astype(object), bdata[:, 2], color="tab:blue", lw=1.0, label="Bz")
    if np.any(focus_bmask):
        b_abs = np.linalg.norm(bdata[focus_bmask], axis=1)
        b_abs_max = float(np.nanpercentile(b_abs[np.isfinite(b_abs)], 99.5)) if np.any(np.isfinite(b_abs)) else np.nan
        b_component_max = float(np.nanmax(np.abs(bdata[focus_bmask])))
        b_lim = max(b_abs_max, b_component_max)
        if np.isfinite(b_lim) and b_lim > 0:
            axes[0].set_ylim(-1.12 * b_lim, 1.12 * b_lim)
    axes[0].set_ylabel("B MSO\n(nT)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].grid(True, ls=":", alpha=0.3)

    panels = [
        ("Bperp PSD", result["Bperp"], "Spectral_r", _positive_norm(result["Bperp"])),
        ("Bperp/Bpara", result["Bperp"] / result["Bpara"], "coolwarm", None),
        ("Theta kB (deg)", result["theta_deg"], "coolwarm", None),
        ("Ellipticity", result["ellipticity"], "coolwarm", None),
        ("Planarity", result["planarity"], "Spectral_r", None),
    ]
    for ax, (label, data, cmap, norm) in zip(axes[1:6], panels):
        pc = ax.pcolormesh(t_num, freq, np.asarray(data).T, shading="auto", cmap=cmap, norm=norm)
        ax.set_yscale("log")
        ax.set_ylabel("Freq\n(Hz)")
        ax.set_ylim(max(freq[0], 0.008), freq[-1])
        ax.grid(True, which="both", ls=":", alpha=0.18)
        ax.text(0.01, 0.86, label, transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        fig.colorbar(pc, ax=ax, pad=0.01)

    if rows:
        centers = np.asarray([row["center"] for row in rows]).astype("datetime64[ms]").astype(object)
        fgyro = np.asarray([row["fgyro_hz"] for row in rows], dtype=float)
        blow = np.asarray([row["band_low_hz"] for row in rows], dtype=float)
        bhigh = np.asarray([row["band_high_hz"] for row in rows], dtype=float)
        pcw = np.asarray([1 if row["is_pcw"] else 0 for row in rows], dtype=float)
        for ax in axes[1:6]:
            ax.plot(centers, fgyro, color="black", lw=2.4, label="fcp", zorder=6)
            ax.plot(centers, blow, color="black", lw=1.5, ls="--", alpha=0.95, zorder=6)
            ax.plot(centers, bhigh, color="black", lw=1.5, ls="--", alpha=0.95, zorder=6)
        axes[1].legend(loc="upper right", fontsize=9, frameon=True)
        axes[6].step(centers, pcw, where="mid", color="tab:red", lw=2.4)
    axes[6].set_ylabel("is_pcw")
    axes[6].set_ylim(-0.1, 1.1)
    axes[6].set_yticks([0, 1])
    axes[6].grid(True, ls=":", alpha=0.35)
    axes[6].set_xlabel("Time")
    axes[6].set_xlim(focus_xlim)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    axes[0].set_title(
        f"{np.datetime_as_string(np.datetime64(focus_start), unit='s')} to "
        f"{np.datetime_as_string(np.datetime64(focus_stop), unit='s')}"
    )
    return fig


def _band_nanmean(values: np.ndarray, freq: np.ndarray, f_low: float, f_high: float) -> float:
    band = (freq >= f_low) & (freq <= f_high)
    if not np.any(band):
        return np.nan
    data = np.asarray(values)[..., band]
    if not np.any(np.isfinite(data)):
        return np.nan
    return float(np.nanmean(data))


def _time_band_nanmean(values: np.ndarray, freq: np.ndarray, f_low: float, f_high: float) -> np.ndarray:
    band = (freq >= f_low) & (freq <= f_high)
    out = np.full(np.asarray(values).shape[0], np.nan, dtype=float)
    if not np.any(band):
        return out
    data = np.asarray(values, dtype=float)[:, band]
    good = np.any(np.isfinite(data), axis=1)
    out[good] = np.nanmean(data[good], axis=1)
    return out


def _longest_true_duration_s(mask: np.ndarray, time: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0
    time = np.asarray(time).astype("datetime64[ns]")
    if time.size < 2:
        return 0.0
    dt_s = np.nanmedian(np.diff(time).astype("timedelta64[ns]").astype(float)) * 1.0e-9
    if not np.isfinite(dt_s) or dt_s <= 0:
        dt_s = 0.0
    best = 0
    current = 0
    for item in mask:
        if item:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best * dt_s)


def evaluate_pcw_svd_psd_criteria_window(
    wave_res: dict[str, Any],
    start: Any,
    stop: Any,
    b0_vec_nt: np.ndarray,
    ion: str | tuple[float, float] = "H+",
    *,
    min_duration_gyroperiods: float = 3.0,
) -> dict[str, Any]:
    """Evaluate the new PCW PSD criteria for one time window using precomputed SVD_B output."""
    start = np.datetime64(start)
    stop = np.datetime64(stop)
    b0_vec = np.asarray(b0_vec_nt, dtype=float)
    b0_abs_nt = float(np.linalg.norm(b0_vec))
    if b0_vec.size != 3 or not np.isfinite(b0_abs_nt) or b0_abs_nt <= 0:
        return {
            "is_pcw": False,
            "reason": "bad_b0",
            "theta_kb_deg": np.nan,
            "ellipticity": np.nan,
            "psd_bperp_0p8_1p2_fci": np.nan,
            "psd_bpara_0p8_1p2_fci": np.nan,
        }

    mass, charge = _ion_mass_charge(ion)
    fci_hz = abs(charge) * b0_abs_nt * 1e-9 / (2.0 * np.pi * mass)
    tci_s = 1.0 / fci_hz if fci_hz > 0 else np.nan

    svd_time = np.asarray(wave_res["Bperp"].time.data)
    smask = (svd_time >= start) & (svd_time < stop)
    freq = np.asarray(wave_res["Bperp"].coords["frequency"].data, dtype=float)
    bperp = np.asarray(wave_res["Bperp"].values, dtype=float)[smask]
    bpara = np.asarray(wave_res["Bpara"].values, dtype=float)[smask]
    theta = np.asarray(wave_res["theta"].values, dtype=float)[smask]
    ellipticity = np.asarray(wave_res["ellipticity"].values, dtype=float)[smask]
    local_time = svd_time[smask]

    low1, high1 = 0.6 * fci_hz, 0.8 * fci_hz
    low_main, high_main = 0.8 * fci_hz, 1.2 * fci_hz
    low2, high2 = 1.2 * fci_hz, 1.4 * fci_hz

    psd_bperp_low = _band_nanmean(bperp, freq, low1, high1)
    psd_bperp_main = _band_nanmean(bperp, freq, low_main, high_main)
    psd_bperp_high = _band_nanmean(bperp, freq, low2, high2)
    psd_bpara_main = _band_nanmean(bpara, freq, low_main, high_main)
    theta_main = _band_nanmean(theta, freq, low_main, high_main)
    ellipticity_main = _band_nanmean(ellipticity, freq, low_main, high_main)

    bperp_low_t = _time_band_nanmean(bperp, freq, low1, high1)
    bperp_main_t = _time_band_nanmean(bperp, freq, low_main, high_main)
    bperp_high_t = _time_band_nanmean(bperp, freq, low2, high2)
    bpara_main_t = _time_band_nanmean(bpara, freq, low_main, high_main)
    theta_t = _time_band_nanmean(theta, freq, low_main, high_main)
    ell_t = _time_band_nanmean(ellipticity, freq, low_main, high_main)

    with np.errstate(divide="ignore", invalid="ignore"):
        low_ratio = psd_bperp_main / psd_bperp_low
        high_ratio = psd_bperp_main / psd_bperp_high
        para_ratio = psd_bperp_main / psd_bpara_main
        low_ratio_t = bperp_main_t / bperp_low_t
        high_ratio_t = bperp_main_t / bperp_high_t
        para_ratio_t = bperp_main_t / bpara_main_t

    good_t = (
        np.isfinite(low_ratio_t)
        & np.isfinite(high_ratio_t)
        & np.isfinite(para_ratio_t)
        & (low_ratio_t >= 1.5)
        & (high_ratio_t >= 1.5)
        & (para_ratio_t >= 2.0)
        & (theta_t <= 45.0)
        & (ell_t < -0.6)
    )
    continuous_duration_s = _longest_true_duration_s(good_t, local_time)
    required_duration_s = float(min_duration_gyroperiods * tci_s) if np.isfinite(tci_s) else np.inf

    psd_ok = bool(
        np.isfinite(low_ratio)
        and np.isfinite(high_ratio)
        and np.isfinite(para_ratio)
        and low_ratio >= 1.5
        and high_ratio >= 1.5
        and para_ratio >= 2.0
    )
    theta_ok = bool(np.isfinite(theta_main) and theta_main <= 45.0)
    ellipticity_ok = bool(np.isfinite(ellipticity_main) and ellipticity_main < -0.6)
    duration_ok = bool(continuous_duration_s >= required_duration_s)
    is_pcw = bool(psd_ok and theta_ok and ellipticity_ok and duration_ok)

    return {
        "is_pcw": is_pcw,
        "is_icw": is_pcw,
        "reason": "pass" if is_pcw else "criteria_not_met",
        "b0_abs_nt": b0_abs_nt,
        "B0x_mso_nT": float(b0_vec[0]),
        "B0y_mso_nT": float(b0_vec[1]),
        "B0z_mso_nT": float(b0_vec[2]),
        "fci_hz": float(fci_hz),
        "Tci_s": float(tci_s),
        "band_low_hz": float(low_main),
        "band_high_hz": float(high_main),
        "theta_kb_deg": float(theta_main) if np.isfinite(theta_main) else np.nan,
        "ellipticity": float(ellipticity_main) if np.isfinite(ellipticity_main) else np.nan,
        "psd_bperp_0p6_0p8_fci": float(psd_bperp_low) if np.isfinite(psd_bperp_low) else np.nan,
        "psd_bperp_0p8_1p2_fci": float(psd_bperp_main) if np.isfinite(psd_bperp_main) else np.nan,
        "psd_bperp_1p2_1p4_fci": float(psd_bperp_high) if np.isfinite(psd_bperp_high) else np.nan,
        "psd_bpara_0p8_1p2_fci": float(psd_bpara_main) if np.isfinite(psd_bpara_main) else np.nan,
        "perp_main_over_low_ratio": float(low_ratio) if np.isfinite(low_ratio) else np.nan,
        "perp_main_over_high_ratio": float(high_ratio) if np.isfinite(high_ratio) else np.nan,
        "perp_over_para_main_ratio": float(para_ratio) if np.isfinite(para_ratio) else np.nan,
        "continuous_duration_s": continuous_duration_s,
        "required_duration_s": required_duration_s,
        "good_svd_samples": int(np.count_nonzero(good_t)),
        "svd_sample_count": int(np.count_nonzero(smask)),
        "psd_ok": psd_ok,
        "theta_ok": theta_ok,
        "ellipticity_ok": ellipticity_ok,
        "duration_ok": duration_ok,
    }


def detect_pcw_svd_psd_criteria(
    Bwave: Any,
    ion: str | tuple[float, float] = "H+",
    *,
    svd_window_length: float = 300.0,
    svd_overlap: float = 150.0,
    m_width_coeff: float = 1,
    nav: int = 12,
    min_duration_gyroperiods: float = 3.0,
) -> dict[str, Any]:
    """
    Detect PCW in one Bwave segment using PSD, theta_kB, ellipticity, and duration criteria.

    The tested bands are 0.6-0.8 fci, 0.8-1.2 fci, and 1.2-1.4 fci.
    """
    time, b_xyz_nt = _extract_bwave_arrays(Bwave)
    if time.size < 2:
        return {
            "is_pcw": False,
            "reason": "too_few_samples",
            "theta_kb_deg": np.nan,
            "ellipticity": np.nan,
            "psd_bperp_0p8_1p2_fci": np.nan,
            "psd_bpara_0p8_1p2_fci": np.nan,
        }

    sample_dt_s = np.nanmedian(np.diff(time.astype("datetime64[ns]")).astype("timedelta64[ns]").astype(float)) * 1.0e-9
    if not np.isfinite(sample_dt_s) or sample_dt_s <= 0:
        sample_dt_s = 0.0
    duration_s = float((time[-1] - time[0]) / np.timedelta64(1, "s")) + float(sample_dt_s)
    if duration_s < svd_window_length:
        return {
            "is_pcw": False,
            "reason": "shorter_than_svd_window_length",
            "duration_s": duration_s,
            "theta_kb_deg": np.nan,
            "ellipticity": np.nan,
            "psd_bperp_0p8_1p2_fci": np.nan,
            "psd_bpara_0p8_1p2_fci": np.nan,
        }

    b0_vec = np.nanmean(b_xyz_nt, axis=0)
    b0_abs_nt = float(np.linalg.norm(b0_vec))
    if not np.isfinite(b0_abs_nt) or b0_abs_nt <= 0:
        return {
            "is_pcw": False,
            "reason": "bad_b0",
            "theta_kb_deg": np.nan,
            "ellipticity": np.nan,
            "psd_bperp_0p8_1p2_fci": np.nan,
            "psd_bpara_0p8_1p2_fci": np.nan,
        }

    mass, charge = _ion_mass_charge(ion)
    fci_hz = abs(charge) * b0_abs_nt * 1e-9 / (2.0 * np.pi * mass)
    tci_s = 1.0 / fci_hz if fci_hz > 0 else np.nan

    try:
        from py_space_zc import method
    except ImportError as exc:
        raise ImportError("py_space_zc is not available, cannot call method.SVD_B") from exc

    freq_min = max(0.5 * fci_hz, 1.0e-4)
    freq_max = max(1.5 * fci_hz, freq_min * 1.2)
    wave_res = method.SVD_B(
        Bwave.dropna(dim="time") if hasattr(Bwave, "dropna") else Bwave,
        window_length=svd_window_length,
        overlap=svd_overlap,
        freq_range=[freq_min, freq_max],
        m_width_coeff=m_width_coeff,
        nav=nav,
    )

    svd_time = np.asarray(wave_res["Bperp"].time.data)
    freq = np.asarray(wave_res["Bperp"].coords["frequency"].data, dtype=float)
    bperp = np.asarray(wave_res["Bperp"].values, dtype=float)
    bpara = np.asarray(wave_res["Bpara"].values, dtype=float)
    theta = np.asarray(wave_res["theta"].values, dtype=float)
    ellipticity = np.asarray(wave_res["ellipticity"].values, dtype=float)

    low1, high1 = 0.6 * fci_hz, 0.8 * fci_hz
    low_main, high_main = 0.8 * fci_hz, 1.2 * fci_hz
    low2, high2 = 1.2 * fci_hz, 1.4 * fci_hz

    psd_bperp_low = _band_nanmean(bperp, freq, low1, high1)
    psd_bperp_main = _band_nanmean(bperp, freq, low_main, high_main)
    psd_bperp_high = _band_nanmean(bperp, freq, low2, high2)
    psd_bpara_main = _band_nanmean(bpara, freq, low_main, high_main)
    theta_main = _band_nanmean(theta, freq, low_main, high_main)
    ellipticity_main = _band_nanmean(ellipticity, freq, low_main, high_main)

    bperp_low_t = _time_band_nanmean(bperp, freq, low1, high1)
    bperp_main_t = _time_band_nanmean(bperp, freq, low_main, high_main)
    bperp_high_t = _time_band_nanmean(bperp, freq, low2, high2)
    bpara_main_t = _time_band_nanmean(bpara, freq, low_main, high_main)
    theta_t = _time_band_nanmean(theta, freq, low_main, high_main)
    ell_t = _time_band_nanmean(ellipticity, freq, low_main, high_main)

    with np.errstate(divide="ignore", invalid="ignore"):
        low_ratio_t = bperp_main_t / bperp_low_t
        high_ratio_t = bperp_main_t / bperp_high_t
        para_ratio_t = bperp_main_t / bpara_main_t
        low_ratio = psd_bperp_main / psd_bperp_low
        high_ratio = psd_bperp_main / psd_bperp_high
        para_ratio = psd_bperp_main / psd_bpara_main

    good_t = (
        np.isfinite(low_ratio_t)
        & np.isfinite(high_ratio_t)
        & np.isfinite(para_ratio_t)
        & (low_ratio_t >= 1.5)
        & (high_ratio_t >= 1.5)
        & (para_ratio_t >= 2.0)
        & (theta_t <= 45.0)
        & (ell_t < -0.6)
    )
    continuous_duration_s = _longest_true_duration_s(good_t, svd_time)
    required_duration_s = float(min_duration_gyroperiods * tci_s) if np.isfinite(tci_s) else np.inf

    psd_ok = bool(
        np.isfinite(low_ratio)
        and np.isfinite(high_ratio)
        and np.isfinite(para_ratio)
        and low_ratio >= 1.5
        and high_ratio >= 1.5
        and para_ratio >= 2.0
    )
    theta_ok = bool(np.isfinite(theta_main) and theta_main <= 45.0)
    ellipticity_ok = bool(np.isfinite(ellipticity_main) and ellipticity_main < -0.6)
    duration_ok = bool(continuous_duration_s >= required_duration_s)
    is_pcw = bool(psd_ok and theta_ok and ellipticity_ok and duration_ok)

    return {
        "is_pcw": is_pcw,
        "is_icw": is_pcw,
        "reason": "pass" if is_pcw else "criteria_not_met",
        "duration_s": duration_s,
        "b0_abs_nt": b0_abs_nt,
        "B0x_mso_nT": float(b0_vec[0]),
        "B0y_mso_nT": float(b0_vec[1]),
        "B0z_mso_nT": float(b0_vec[2]),
        "fci_hz": float(fci_hz),
        "Tci_s": float(tci_s),
        "band_low_hz": float(low_main),
        "band_high_hz": float(high_main),
        "theta_kb_deg": float(theta_main) if np.isfinite(theta_main) else np.nan,
        "ellipticity": float(ellipticity_main) if np.isfinite(ellipticity_main) else np.nan,
        "psd_bperp_0p6_0p8_fci": float(psd_bperp_low) if np.isfinite(psd_bperp_low) else np.nan,
        "psd_bperp_0p8_1p2_fci": float(psd_bperp_main) if np.isfinite(psd_bperp_main) else np.nan,
        "psd_bperp_1p2_1p4_fci": float(psd_bperp_high) if np.isfinite(psd_bperp_high) else np.nan,
        "psd_bpara_0p8_1p2_fci": float(psd_bpara_main) if np.isfinite(psd_bpara_main) else np.nan,
        "perp_main_over_low_ratio": float(low_ratio) if np.isfinite(low_ratio) else np.nan,
        "perp_main_over_high_ratio": float(high_ratio) if np.isfinite(high_ratio) else np.nan,
        "perp_over_para_main_ratio": float(para_ratio) if np.isfinite(para_ratio) else np.nan,
        "continuous_duration_s": continuous_duration_s,
        "required_duration_s": required_duration_s,
        "good_svd_samples": int(np.count_nonzero(good_t)),
        "svd_sample_count": int(good_t.size),
        "psd_ok": psd_ok,
        "theta_ok": theta_ok,
        "ellipticity_ok": ellipticity_ok,
        "duration_ok": duration_ok,
        "time": svd_time,
        "freqs_hz": freq,
        "wave_res": wave_res,
    }


__all__ = [
    "evaluate_pcw_svd_psd_criteria_window",
    "detect_pcw_svd_psd_criteria",
    "read_sw_interval_list",
]

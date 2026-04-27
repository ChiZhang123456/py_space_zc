import numpy as np
import xarray as xr
from scipy import signal

def filt(inp, f_min: float = 0.0, f_max: float = 0.0, order: int = 4):
    """
    Apply a stable digital filter using Second-Order Sections (SOS).
    Supports Bandpass, Highpass, and Lowpass filters.

    Parameters
    ----------
    inp : xarray.DataArray
        Input time series data. Must contain a 'time' coordinate.
    f_min : float, optional
        Lower cutoff frequency (Hz). If 0, performs a Lowpass filter.
    f_max : float, optional
        Upper cutoff frequency (Hz). If 0, performs a Highpass filter.
    order : int, optional
        Filter order. Default is 4 (equivalent to 8 after filtfilt).

    Returns
    -------
    out : xarray.DataArray
        Filtered time series with zero phase shift.
    """

    # 1. Calculate sampling frequency (fs) and Nyquist frequency
    # Compute the median time delta to be robust against occasional gaps
    time_diff = np.diff(inp.time.data).astype("timedelta64[ns]").astype(float)
    dt = np.median(time_diff) * 1e-9  # Convert nanoseconds to seconds
    fs = 1.0 / dt
    f_nyq = fs / 2.0  # Nyquist limit

    # 2. Validate frequency limits against data resolution
    if f_max > f_nyq:
        raise ValueError(
            f"f_max ({f_max} Hz) exceeds the Nyquist frequency ({f_nyq:.2f} Hz). "
            f"Resolution is too low for this filter."
        )

    # 3. Normalize frequencies to the Nyquist frequency (0 to 1 range)
    low = f_min / f_nyq
    high = f_max / f_nyq

    # 4. Design the filter based on provided bounds
    # Using Butterworth for maximally flat passband and better stability
    if f_min > 0 and f_max > 0:
        # Bandpass filter
        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    elif f_min > 0:
        # Highpass filter
        sos = signal.butter(order, low, btype='highpass', output='sos')
    elif f_max > 0:
        # Lowpass filter
        sos = signal.butter(order, high, btype='lowpass', output='sos')
    else:
        # No frequencies provided, return original data
        return inp

    # 5. Execute zero-phase filtering
    # axis=0 ensures it works on the time dimension regardless of 1D (scalar) or 2D (vector)
    filtered_data = signal.sosfiltfilt(sos, inp.data, axis=0)

    # 6. Reconstruct xarray.DataArray with original metadata
    out = xr.DataArray(
        filtered_data,
        coords=inp.coords,
        dims=inp.dims,
        attrs=inp.attrs.copy()
    )

    # Update attributes for traceability
    out.attrs["filter_info"] = f"Butterworth SOS {f_min}-{f_max} Hz"
    out.attrs["sampling_rate"] = f"{fs:.2f} Hz"

    return out
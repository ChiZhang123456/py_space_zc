import numpy as np
from scipy.signal import savgol_filter


def smooth(data, smooth_number=11, polyorder=2):
    """
    Smooth 1D or 2D data using the Savitzky-Golay filter.

    This function applies the Savitzky-Golay smoothing algorithm, which
    performs a local polynomial regression on a series of adjacent data
    points. It is particularly effective at preserving the overall shape
    and features of the signal (such as peaks) while reducing noise,
    making it superior to a simple moving average in many cases.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be:
        - 1D array of shape (n,)
        - 2D array of shape (n, m), e.g., (n,2), (n,3)
        The smoothing is applied independently to each column if the input
        is 2D or higher.

    smooth_number : int, optional (default=11)
        Window length for smoothing. This must be:
        - An odd integer (required by Savitzky-Golay filter for symmetry)
        - Less than or equal to the length of the input data
        If an invalid value is provided, the function will automatically
        adjust it (e.g., make it odd or reduce it if larger than n).

    polyorder : int, optional (default=2)
        The order of the polynomial used for fitting within each window.
        Must be less than `smooth_number`.

    Returns
    -------
    smoothed : np.ndarray
        The smoothed array with the same shape as the input data.
        - For 1D input (n,), output is 1D (n,)
        - For 2D input (n,m), output is 2D (n,m)

    Notes
    -----
    - Savitzky-Golay smoothing is more advanced than a moving average: it
      reduces noise but better preserves edges, peaks, and the overall
      signal structure.
    - For very short datasets, set `smooth_number` small (>=3).
    - If `smooth_number` is even, the function will automatically
      increase it to the next odd number.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 2*np.pi, 50)
    >>> y = np.sin(x) + 0.2*np.random.randn(50)   # noisy sine wave
    >>> y_smooth = smooth(y, smooth_number=7, polyorder=2)

    >>> # 2D example: smooth two columns independently
    >>> data = np.column_stack((y, y*2))
    >>> data_smooth = smooth(data, smooth_number=7)
    """
    data = np.asarray(data)

    # Ensure the window length (smooth_number) is valid
    n = data.shape[0]
    if smooth_number > n:
        # If the window is larger than the data length, shrink it
        smooth_number = n if n % 2 == 1 else n - 1
    if smooth_number < 3:
        # Minimum valid window length is 3
        smooth_number = 3
    if smooth_number % 2 == 0:
        # Window length must be odd for Savitzky-Golay
        smooth_number += 1

    # Case 1: 1D data
    if data.ndim == 1:
        return savgol_filter(data, window_length=smooth_number, polyorder=polyorder)

    # Case 2: Multi-dimensional data
    # Apply smoothing independently on each column
    smoothed = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        smoothed[:, i] = savgol_filter(data[:, i],
                                       window_length=smooth_number,
                                       polyorder=polyorder)
    return smoothed

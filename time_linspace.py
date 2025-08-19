import numpy as np

def time_linspace(start_time, end_time, number=5, unit='ms'):
    """
    Generate an equally spaced array of datetime64 values between start_time and end_time.

    Parameters
    ----------
    start_time, end_time : str | pandas.Timestamp | numpy.datetime64
        The start and end times. Can be given as strings (e.g., "2017-01-01T00:00:00"),
        pandas Timestamps, or NumPy datetime64 objects.
    number : int, default 5
        Number of time points to generate, including start and end.
    unit : {'ns','us','ms','s','m','h','D'}, default 'ms'
        Time resolution unit for calculations. The output will also use this resolution.

    Returns
    -------
    numpy.ndarray(dtype='datetime64[unit]')
        Array of equally spaced time points.

    Examples
    --------
    >>> time_linspace("2017-01-01T00:00:00", "2017-01-01T00:10:00", 11)
    array([...], dtype='datetime64[ms]')
    """

    if number <= 0:
        raise ValueError("`number` must be a positive integer.")

    # Convert inputs to numpy.datetime64 with the desired resolution
    start = np.datetime64(start_time, unit)
    end = np.datetime64(end_time, unit)

    # Special case: only one point requested → return start only
    if number == 1:
        return np.array([start], dtype=f'datetime64[{unit}]')

    # Convert datetimes to integer timestamps (number of 'unit' ticks since epoch)
    start_i = start.astype('int64')
    end_i = end.astype('int64')

    # Perform linear spacing on integer timestamps
    # Use np.rint to avoid floating-point precision errors before casting back to int64
    ticks_i = np.rint(np.linspace(start_i, end_i, number)).astype('int64')

    # Convert integer ticks back to datetime64 with the same unit
    return ticks_i.astype(f'datetime64[{unit}]')

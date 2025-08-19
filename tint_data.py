import numpy as np

def tint_data(time: np.ndarray, ts: np.datetime64, te: np.datetime64, *datas):
    """
    Extract time and multiple data arrays within time interval [ts, te].

    Author: Chi Zhang

    Parameters
    ----------
    time : np.ndarray
        1D array of datetime64[ns], shape: (n_time,)
    ts : np.datetime64
        Start time
    te : np.datetime64
        End time
    *datas : one or more np.ndarray
        Data arrays, each must have shape (n_time, ...)

    Returns
    -------
    time_new : np.ndarray
        Time array within [ts, te]
    data_new_list : tuple of np.ndarray
        Tuple of sliced data arrays corresponding to the input datas
    """
    assert isinstance(time, np.ndarray) and time.ndim == 1, "time must be 1D numpy array"
    assert isinstance(ts, np.datetime64) and isinstance(te, np.datetime64), "ts and te must be datetime64"
    for i, d in enumerate(datas):
        assert isinstance(d, np.ndarray), f"data[{i}] must be numpy array"
        assert d.shape[0] == time.shape[0], f"data[{i}] must match time dimension"

    mask = (time >= ts) & (time <= te)
    time_new = time[mask]
    data_new_list = tuple(d[mask, ...] for d in datas)

    return (time_new, *data_new_list)

if __name__ == '__main__':
    # 假设你有如下输入：
    time = np.array(['2022-01-01T00:00:00', '2022-01-01T00:01:00', '2022-01-01T00:02:00'], dtype='datetime64[ns]')
    data1 = np.random.rand(3, 48)
    data2 = np.random.rand(3, 48, 16)

    ts = np.datetime64('2022-01-01T00:00:30')
    te = np.datetime64('2022-01-01T00:01:30')

    # 调用函数
    time_new, data1_new, data2_new = tint_data(time, ts, te, data1, data2)

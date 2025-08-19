import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
from .irf_time import irf_time

def adjust_time(time, seconds):
    """
    Adjusts the input time(s) by a given number of seconds.
    Supports various time formats: str, datetime64, datetime, and MATLAB datenum.
    Can handle both single time inputs and arrays of times.

    Args:
        time: Input time(s) in str, datetime64, datetime, or MATLAB datenum format.
              Can be a single value or a numpy array.
        seconds (int): Number of seconds to adjust, can be negative or positive.

    Returns:
        Adjusted time(s) in the same format as the input.
    """
    # 检查是否为数组输入
    is_array = isinstance(time, (list, np.ndarray))
    if is_array:
        time = np.array(time)
    else:
        time = np.array([time])

    # 确定输入时间的格式
    if time.dtype.kind == 'U' or time.dtype.kind == 'S':
        input_format = 'iso8601'
        parsed_time = np.array([parse(t) for t in time])
    elif time.dtype.kind == 'M':
        input_format = 'datetime64'
        parsed_time = time
    elif time.dtype.kind == 'O' and isinstance(time[0], datetime):
        input_format = 'datetime'
        parsed_time = np.array([np.datetime64(t) for t in time])
    elif np.issubdtype(time.dtype, np.number):
        input_format = 'datenum'
        parsed_time = irf_time(time, 'datenum>datetime64')
    else:
        raise ValueError("Unsupported time format")

    # 执行时间调整
    adjusted_time = parsed_time.astype('datetime64[s]') + np.timedelta64(seconds, 's')
    new_time = irf_time(adjusted_time, "datetime64>datetime")
    
    # 根据输入格式返回结果
    if input_format == 'iso8601':
        if np.any([("." in str(t)) for t in time]):  # 检查是否有输入包含毫秒
            result = np.array([t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] for t in new_time])
        else:
            result = np.array([t.strftime("%Y-%m-%dT%H:%M:%S") for t in new_time])
    elif input_format == 'datetime64':
        result = irf_time(new_time , "datetime>datetime64")
    elif input_format == 'datetime':
        result = new_time
    elif input_format == 'datenum':
        result = irf_time(new_time, 'datetime>datenum')

    # 如果输入是单个值，返回单个结果；否则返回数组
    return result[0] if not is_array else result

#%% 使用示例
if __name__ == "__main__":
    # 示例 1: 字符串格式数组
    str_times = np.array(["2023-01-01T12:00:00", "2023-01-02T12:00:00"])
    print("String time array adjustment:")
    print(f"Original: {str_times}")
    print(f"Adjusted: {adjust_time(str_times, 3600)}")  # 每个时间增加1小时

    # 示例 2: datetime64 格式数组
    dt64_times = np.array(['2023-01-01T12:00:00', '2023-01-02T12:00:00'], dtype='datetime64')
    print("\nDatetime64 time array adjustment:")
    print(f"Original: {dt64_times}")
    print(f"Adjusted: {adjust_time(dt64_times, -1800)}")  # 每个时间减少30分钟

    # 示例 3: datetime 格式数组
    dt_times = np.array([datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 2, 12, 0, 0)])
    print("\nDatetime object array adjustment:")
    print(f"Original: {dt_times}")
    print(f"Adjusted: {adjust_time(dt_times, 86400)}")  # 每个时间增加1天

    # 示例 4: MATLAB datenum 格式数组
    datenum_times = np.array([738885.5, 738886.5])  # 2023-01-01 12:00:00 和 2023-01-02 12:00:00
    print("\nMATLAB datenum time array adjustment:")
    print(f"Original: {datenum_times}")
    print(f"Adjusted: {adjust_time(datenum_times, 43200)}")  # 每个时间增加12小时

    # 示例 5: 单个时间输入
    single_time = datetime(2023, 1, 1, 12, 0, 0)
    print("\nSingle datetime object adjustment:")
    print(f"Original: {single_time}")
    print(f"Adjusted: {adjust_time(single_time, 3600)}")  # 增加1小时
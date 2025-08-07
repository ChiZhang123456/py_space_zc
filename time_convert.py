# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:52:16 2024

@author: Win
"""

import numpy as np
import pandas as pd
import spiceypy as spice
from datetime import datetime
from cdflib import cdfepoch
import re

#%%
def datenum2datetime64(matlab_datenum):
    """
    将MATLAB的datenum格式转换为numpy的datetime64[ns]格式
    
    参数:
    matlab_datenum : float, int, list, 或 numpy数组
        MATLAB格式的日期数值
    
    返回:
    numpy.datetime64[ns] 或 包含datetime64[ns]的numpy数组
    """
    
    # 确保输入是numpy数组
    matlab_datenum = np.atleast_1d(np.array(matlab_datenum, dtype=float))
    
    # MATLAB datenum的起始日期是0000年1月1日，而Python的datetime是从1年1月1日开始
    # 因此需要减去719529天来调整这个差异
    days_from_matlab_base = matlab_datenum - 719529
    
    # 转换为timedelta（以纳秒为单位）
    seconds_from_base = days_from_matlab_base * 24 * 3600
    nanoseconds_from_base = seconds_from_base * 1e9
    
    # 使用astype直接转换为timedelta64[ns]
    timedelta_from_base = nanoseconds_from_base.astype('timedelta64[ns]')
    
    # 1970-01-01 是numpy datetime64的起始日期
    numpy_base = np.datetime64('1970-01-01')
    
    # 计算最终的datetime64
    result = numpy_base + timedelta_from_base

    # 如果输入是标量，返回标量结果
    if result.size == 1:
        return result[0]
    return result


#%%
def datetime642datenum(datetime64_arr):
    """
    将numpy的datetime64[ns]格式转换为MATLAB的datenum格式
    
    参数:
    datetime64_arr : numpy.datetime64[ns], 或包含datetime64[ns]的numpy数组
    
    返回:
    float 或 包含float的numpy数组,表示MATLAB格式的日期数值
    """
    
    # 确保输入是numpy数组
    datetime64_arr = np.atleast_1d(datetime64_arr)
    
    # 将datetime64[ns]转换为自1970-01-01以来的纳秒数
    nanoseconds_from_1970 = datetime64_arr.astype('datetime64[ns]').astype(np.int64)
    
    # 转换为天数
    days_from_1970 = nanoseconds_from_1970 / (24 * 3600 * 1e9)
    
    # MATLAB datenum的起始日期是0000年1月1日，而1970-01-01在MATLAB中的datenum是719529
    matlab_datenum = days_from_1970 + 719529
    
    # 如果输入是标量，返回标量结果
    if matlab_datenum.size == 1:
        return matlab_datenum[0]
    return matlab_datenum

#%%
def datetime642datetime(dt64_arr):
    return pd.to_datetime(dt64_arr).to_pydatetime()

#%%
def datetime2datetime64(dt):
    return np.array([np.datetime64(t) for t in dt])

#%%
def datetime2iso8601(time):
    r"""Transforms datetime to TT2000 string format.

    Parameters
    ----------
    time : datetime.datetime
        Time to convert to tt2000 string.

    Returns
    -------
    tt2000 : str
        Time in TT20000 iso_8601 format.

    """

    # Check input type
    message = "time must be array_like or datetime.datetime"
    assert isinstance(time, (list, np.ndarray, datetime.datetime)), message

    if isinstance(time, (np.ndarray, list)):
        return list(map(datetime2iso8601, time))

    assert isinstance(time, datetime.datetime), "time datetime.datetime"

    time_datetime = pd.Timestamp(time)

    # Convert to string
    datetime_str = time_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")

    time_iso8601 = f"{datetime_str}{time_datetime.nanosecond:03d}"

    return time_iso8601

#%%
def iso86012datetime(time):
    r"""Converts ISO 8601 time to datetime

    Parameters
    ----------
    time : ndarray or list or str
        Time

    Returns
    -------
    time_datetime : list
        Time in datetime format.

    """

    # Make sure that str is in ISO8601 format
    time = np.atleast_1d(time).astype("datetime64[ns]").astype(str)

    # ISO 8601 format with miliseconds precision (max precision for datetime)
    fmt = "%Y-%m-%dT%H:%M:%S.%f"

    # Convert to datetime format
    time_datetime = [datetime.datetime.strptime(t_[:-3], fmt) for t_ in time]

    return time_datetime

#%%
def datetime642iso8601(time):
    r"""Convert datetime64 in ns units to ISO 8601 time format .

    Parameters
    ----------
    time : ndarray
        Time in datetime64 in ns units.

    Returns
    -------
    time_iso8601 : ndarray
        Time in ISO 8601 format.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """
    
    if isinstance(time, np.datetime64):
        time = np.array([time])
        time_datetime64 = time.astype("datetime64[ns]")
    elif isinstance(time, (list, np.ndarray)):
        time_datetime64 = time.astype("datetime64[ns]")
    else:
        raise TypeError("time must be numpy.datetime64 or array_like")

    # Convert to string
    time_iso8601 = time_datetime64.astype(str)
    time_iso8601 = np.atleast_1d(np.squeeze(np.stack([time_iso8601])))

    return time_iso8601



#%%

def iso86012datetime64(time):
    r"""Convert ISO8601 time format to datetime64 in ns units.

    Parameters
    ----------
    time : ndarray
        Time in ISO 8601 format

    Returns
    -------
    time_datetime64 : ndarray
        Time in datetime64 in ns units.

    See Also
    --------
    pyrfu.pyrf.datetime642iso8601

    """

    time_datetime64 = time.astype("datetime64[ns]")

    return time_datetime64

#%%
def datetime642et(dt64):
    # 确保输入是 numpy datetime64 类型
    if not isinstance(dt64, (np.datetime64, np.ndarray)):
        raise TypeError("Input must be numpy.datetime64 or numpy.ndarray of datetime64")
    
    # 如果输入是单个 datetime64,将其转换为数组
    if isinstance(dt64, np.datetime64):
        dt64 = np.array([dt64])
    
    # 将 datetime64 转换为 datetime
    dt = dt64.astype(datetime)
    
    # 将 datetime 转换为 ISO 格式的字符串
    iso_strings = [d.strftime("%Y-%m-%dT%H:%M:%S.%f") for d in dt]
    
    # 使用 spice.str2et 将 ISO 字符串转换为 ET
    et = np.array([spice.str2et(iso_str) for iso_str in iso_strings])
    
    # 如果输入是单个值,返回标量;否则返回数组
    return et[0] if len(et) == 1 else et

#%%
def datenum2iso8601(matlab_datenum):
    return datetime642iso8601(datenum2datetime64(matlab_datenum))
#%%
def iso86012datenum(time):
    return datetime642datenum(iso86012datetime64(time))
#%% 
def datenum2datetime(matlab_datenum):
    return datetime642datetime(datenum2datetime64(matlab_datenum))
#%% 
def datetime2datenum(time):
    return datetime642datenum(datetime2datetime64(time))
#%% 
def datenum2et(matlab_datenum):
    return datetime642et(datenum2datetime64(matlab_datenum))
#%%
def datetime642ttns(time):
    r"""Converts datetime64 in ns units to epoch_tt2000
    (nanoseconds since J2000).

    Parameters
    ----------
    time : ndarray
        Times in datetime64 format.

    Returns
    -------
    time_ttns : ndarray
        Times in epoch_tt2000 format (nanoseconds since J2000).

    """

    # Convert to datetime64 in ns units
    time_iso8601 = datetime642iso8601(time)

    # Convert to ttns
    time_ttns = np.array([cdfepoch.parse(t_) for t_ in time_iso8601])

    return time_ttns

#%%
def datetime642unix(time):
    r"""Converts datetime64 in ns units to unix time.

    Parameters
    ----------
    time : ndarray
        Time in datetime64 format.

    Returns
    -------
    time_unix : ndarray
        Time in unix format.

    See Also
    --------
    pyrfu.pyrf.unix2datetime64

    """

    # Make sure that time is in ns format
    if isinstance(time, np.datetime64):
        time = np.array([time])
        time_datetime64 = time.astype("datetime64[ns]")
    elif isinstance(time, (list, np.ndarray)) and isinstance(time[0], np.datetime64):
        time_datetime64 = time.astype("datetime64[ns]")
    else:
        raise TypeError("time must be numpy.datetime64 or array_like")

    # Convert to unix
    time_unix = time_datetime64.astype(np.int64) / 1e9

    return time_unix
#%%
def iso86012timevec(time):
    r"""Convert ISO 8601 time string into time vector.

    Parameters
    ----------
    time : ndarray or list or str
        Time in ISO 8601 format YYYY-MM-DDThh:mm:ss.mmmuuunnn.

    Returns
    -------
    time_vec : list
        Time vector.

    See Also
    --------
    pyrfu.pyrf.iso86012timevec

    """

    iso_8601 = (
        r"(?P<years>[0-9]{4})-(?P<months>[0-9]{2})-(?P<days>[0-9]{2})"
        r"T(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2})"
        r":(?P<seconds>[0-9]{2}).(?P<miliseconds>[0-9]{3})"
        r"(?P<microseconds>[0-9]{3})(?P<nanoseconds>[0-9]{3})"
    )

    # Define parser
    fmt = re.compile(iso_8601)

    # Make time is a 1d array
    time = np.atleast_1d(time)

    time_vec = [[int(p_) for p_ in fmt.match(t_).groups()] for t_ in time]
    time_vec = np.array(time_vec)

    return time_vec


#%%
def timevec2iso8601(time):
    r"""Convert time vector into ISO 8601 format YYYY-MM-DDThh:mm:ss.mmmuuunnn.

    Parameters
    ----------
    time : ndarray
        Time vector

    Returns
    -------
    time_iso8601 : ndarray
        Time in ISO 8601 format.

    See Also
    --------
    pyrfu.pyrf.iso86012timevec

    """

    time = np.atleast_2d(np.array(time))
    time = np.hstack([time, np.zeros((len(time), 9 - time.shape[1]))])

    time_iso8601 = []

    for t_ in time.astype(np.int64):
        ye_mo_da_ = f"{t_[0]:04}-{t_[1]:02}-{t_[2]:02}"  # YYYY-MM-DD
        ho_mi_se_ = f"{t_[3]:02}:{t_[4]:02}:{t_[5]:02}"  # hh:mm:ss
        ms_us_ns_ = f"{t_[6]:03}{t_[7]:03}{t_[8]:03}"  # mmmuuunnn

        # Time as ISO 8601 string 'YYYY-MM-DDThh:mm:ss.mmmuuunnn'
        time_iso8601.append(f"{ye_mo_da_}T{ho_mi_se_}.{ms_us_ns_}")

    time_iso8601 = np.array(time_iso8601)

    return time_iso8601

#%%
def iso86012unix(time):
    r"""Converts time in iso format to unix

    Parameters
    ----------
    time : str or array_like of str
        Time.

    Returns
    -------
    out : float or list of float
        Time in unix format.

    """

    assert isinstance(time, (str, list, np.ndarray)), "time must be a str or array_like"

    out = np.atleast_1d(time).astype("datetime64[ns]")

    return out

#%% 示例用法
if __name__ == "__main__":
    # 示例：datenum 到 datetime64 的转换
    datenum_example = 737967.4583333334  # 对应 2020-06-25 11:00
    result = datenum2datetime64(datenum_example)
    print(f"Datenum to Datetime64: {result}")

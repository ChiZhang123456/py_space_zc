# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:02:43 2024

@author: Win
"""
from .time_convert import datenum2datetime64, datetime642datenum, datetime2datetime64
from .time_convert import datetime642datetime
from .time_convert import iso86012datetime64
from .time_convert import datetime642iso8601
from .time_convert import datenum2iso8601
from .time_convert import iso86012datenum
from .time_convert import datetime642et
from .time_convert import datenum2et
from .time_convert import datetime642ttns
from .time_convert import datetime642unix
from .time_convert import timevec2iso8601
from .time_convert import iso86012timevec
from .time_convert import datetime2datenum
from .time_convert import datenum2datetime
        
def irf_time(time, coords:str):
    """
    通用时间转换函数

    参数:
    time : 输入的时间，可以是各种格式
    coords : str, 指定转换的起始和目标格式，格式为 "from>to"

    返回:
    转换后的时间

    支持的格式:
    - datenum
    - datetime64
    - datetime
    - iso8601
    - et (Ephemeris Time)
    - ttns (TT2000 nanoseconds)
    - unix
    - timevec
    """
    
    from_format, to_format = coords.split('>')
    
    # 定义转换函数字典
    converters = {
        "datenum>datetime64": datenum2datetime64,
        "datetime64>datenum": datetime642datenum,
        "datetime>datetime64": datetime2datetime64,
        "datetime64>datetime": datetime642datetime,
        "datetime>datenum":datetime2datenum,
        "datenum>datetime":datenum2datetime,
        "iso8601>datetime64": iso86012datetime64,
        "datetime64>iso8601": datetime642iso8601,
        "datenum>iso8601": datenum2iso8601,
        "iso8601>datenum": iso86012datenum,
        "datetime64>et": datetime642et,
        "datenum>et": datenum2et,
        "datetime64>ttns": datetime642ttns,
        "datetime64>unix": datetime642unix,
        "timevec>iso8601": timevec2iso8601,
        "iso8601>timevec": iso86012timevec,
    }
    
    # 如果直接转换函数存在，直接使用
    if coords in converters:
        return converters[coords](time)
    
    # 否则，尝试通过 datetime64 进行中间转换
    to_datetime64 = converters.get(f"{from_format}>datetime64")
    from_datetime64 = converters.get(f"datetime64>{to_format}")
    
    if to_datetime64 and from_datetime64:
        intermediate = to_datetime64(time)
        return from_datetime64(intermediate)
    
    raise ValueError(f"Unsupported conversion: {coords}")
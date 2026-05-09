# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:02:43 2024

Author: Chi Zhang
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
    General time conversion helper.

    Parameters
    ----------
    time
        Input time in one of the supported formats.
    coords : str
        Conversion direction in the form "from>to".

    Returns
    -------
    Converted time value.

    Supported formats
    -----------------
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
    
    # Map explicit conversion paths to their implementation functions.
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
    
    # Use a direct converter when one is available.
    if coords in converters:
        return converters[coords](time)
    
    # Otherwise, try datetime64 as an intermediate representation.
    to_datetime64 = converters.get(f"{from_format}>datetime64")
    from_datetime64 = converters.get(f"datetime64>{to_format}")
    
    if to_datetime64 and from_datetime64:
        intermediate = to_datetime64(time)
        return from_datetime64(intermediate)
    
    raise ValueError(f"Unsupported conversion: {coords}")

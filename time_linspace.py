# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:15:01 2024

@author: Win
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse

#%%
def time_linspace(start_time, end_time, number=5):
    """
    Generates an equally spaced time series within the given time range.

    Parameters:
    - start_time (numpy.datetime64): The start time of the time series.
    - end_time (numpy.datetime64): The end time of the time series.
    - number (int, optional): The number of time points to generate, default is 5.

    Returns:
    - numpy.ndarray: An array containing equally spaced time points, type datetime64[ms].

    Examples:
       start_time = np.datetime64("2017-01-01T00:00:00.000")
       end_time = np.datetime64("2017-01-01T00:10:00.000")
       time_arr = time_linspace(start_time, end_time, 11)

    Author:
    Chi Zhang
    """
    # convert to float type
    start_timestamp = start_time.astype('float64')
    end_timestamp = end_time.astype('float64')
    timestamps = np.linspace(start_timestamp, end_timestamp, num=number).astype('float64')
   
    # back to datetime64 type
    tarr = timestamps.astype('datetime64[ms]')
    return tarr
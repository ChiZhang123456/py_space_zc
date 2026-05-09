# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:15:32 2024

Author: Chi Zhang
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse



#%%
def year_month_day(time_datetime64):
    # Convert a numpy datetime64 object to a native Python datetime object.
    time_datetime = time_datetime64.astype('O')
    year = time_datetime.strftime('%Y')
    month = time_datetime.strftime('%m')
    day = time_datetime.strftime('%d')
    return year, month, day

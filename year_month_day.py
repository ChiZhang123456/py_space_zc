# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:15:32 2024

@author: Win
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse



#%%
def year_month_day(time_datetime64):
    # 直接将 datetime64 对象转换为 Python 的 datetime.date 对象
    time_datetime = time_datetime64.astype('O')  # 'O' 将 numpy datetime64 转换为原生 Python datetime
    year = time_datetime.strftime('%Y')
    month = time_datetime.strftime('%m')
    day = time_datetime.strftime('%d')
    return year, month, day
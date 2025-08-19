# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:03:43 2024

@author: Win
"""
import os
from datetime import datetime, timedelta
import py_space_zc.emm as emm
from dateutil import parser

def find_closest_file(input_time, mode='osr'):
    # 基础目录路径设置
    base_directory = os.path.join(emm.get_base_path(), "emu", "l2a")    
    # 解析输入时间并构建路径
    if isinstance(input_time, str):
        input_datetime = parser.parse(input_time)
        year_month_path = os.path.join(base_directory, mode, str(input_datetime.year), f"{input_datetime.month:02}")
        return search_for_file(input_datetime, year_month_path)
    elif isinstance(input_time, list):
        result_files = []
        for time in input_time:
            input_datetime = parser.parse(time)
            year_month_path = os.path.join(base_directory, mode, str(input_datetime.year), f"{input_datetime.month:02}")
            result_files.append(search_for_file(input_datetime, year_month_path))
        return result_files


def search_for_file(input_time, directory):
    smallest_diff = timedelta.max
    closest_file = None

    # 检查目录是否存在
    if not os.path.exists(directory):
        return None  # 如果目录不存在，返回 None

    # 遍历文件夹中的每个文件
    for file in os.listdir(directory):
        if file.endswith(".fits.gz"):
            # 提取文件名中的时间戳
            timestamp_str = file.split('_')[3]  # 根据文件名结构调整索引
            file_time = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
            
            # 计算时间差
            diff = abs(file_time - input_time)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_file = os.path.join(directory, file)
    
    return closest_file

# 示例用法
if __name__ == "__main__":
    base_directory = 'F:\\data\\emm\\data\\emu\\l2a'
    filelist = find_closest_file(['2022-09-04T01:58','2022-09-04T01:58'], base_directory)


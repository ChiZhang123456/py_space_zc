# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:03:43 2024

Author: Chi Zhang
"""
import os
from datetime import datetime, timedelta
import py_space_zc.emm as emm
from dateutil import parser

def find_closest_file(input_time, mode='osr'):
    # Configure the base data directory.
    base_directory = os.path.join(emm.get_base_path(), "emu", "l2a")    
    # Parse the input time and build the year/month data path.
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

    # Check whether the directory exists.
    if not os.path.exists(directory):
        return None

    # Search each file in the directory.
    for file in os.listdir(directory):
        if file.endswith(".fits.gz"):
            # Extract the timestamp from the filename.
            timestamp_str = file.split('_')[3]
            file_time = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
            
            # Compute the absolute time difference.
            diff = abs(file_time - input_time)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_file = os.path.join(directory, file)
    
    return closest_file

# Example usage
if __name__ == "__main__":
    base_directory = 'F:\\data\\emm\\data\\emu\\l2a'
    filelist = find_closest_file(['2022-09-04T01:58','2022-09-04T01:58'], base_directory)


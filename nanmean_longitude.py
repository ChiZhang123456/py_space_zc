# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:37:19 2024

@author: Win
"""

import numpy as np

def nanmean_longitude(lon):
    """
    计算二维矩阵中的经度数据的环形平均值。
    
    参数:
    lon -- 二维经度矩阵 (单位：度)
    
    返回:
    circ_mean_lon -- 经度的环形平均值
    """
    # 将经度转换为弧度
    lon_rad = np.radians(lon)
    
    # 计算经度向量的余弦和正弦
    cos_lon = np.nanmean(np.cos(lon_rad), axis=(0, 1))
    sin_lon = np.nanmean(np.sin(lon_rad), axis=(0, 1))
    
    # 计算平均向量的角度
    circ_mean_lon_rad = np.arctan2(sin_lon, cos_lon)
    
    # 将结果转换回度
    circ_mean_lon = np.degrees(circ_mean_lon_rad)
    
    return circ_mean_lon

if __name__ == "__main__":
    lon = np.array([[1, 359], [2, 358], [np.nan, 3]])  # 示例数据
    mean_longitude = nanmean_longitude(lon)
    print("环形平均经度为：", mean_longitude)

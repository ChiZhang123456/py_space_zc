# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:51:02 2024

@author: Win
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from py_space_zc.maven import bs_mpb
from py_space_zc.maven import get_data

def show_path_xr(ax, Pmso):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    
    time = Pmso.time.data
    data = Pmso.values
    X = data[:,0] / 3390
    R = np.sqrt(data[:,1]**2 + data[:,2]**2) / 3390
    
    # 将 datetime64 时间转换为 matplotlib 内部使用的数值格式
    time_in_mpl = mdates.date2num(time.astype('datetime64[s]').astype('O'))

    bs_mpb(ax)
    scatter = ax.scatter(X, R, c=time_in_mpl, s=15, cmap="Spectral_r", edgecolors='None')
    
    ax.set_xlabel(r"$X_{\mathrm{MSO}}$ (Rm)")
    ax.set_ylabel(r"$\sqrt{Y_{\mathrm{MSO}}^2 + Z_{\mathrm{MSO}}^2}$ (Rm)")

    # 添加颜色条到图的外侧
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.04, pad=0.005, aspect=14)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 设置时间格式为时分秒
    #cbar.set_label('Time', labelpad=-50, rotation=270)




# 示例用法
# 假设 Pmso 已经是一个适当配置的 xarray.DataArray 或类似的包含时间的对象
# show_path_xr(None, Pmso)

 
if __name__ == "__main__":
    tint = ["2022-09-23T13:25", "2022-09-23T13:30"]
    B = get_data(tint, "B")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)  # right=0.9 表示右侧留出10%的空白
    show_path_xr(ax, B["Pmso"])

    
    
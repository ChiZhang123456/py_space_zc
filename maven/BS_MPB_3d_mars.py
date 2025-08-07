import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def BS_MPB_3d_mars(ax):
    # 绘制弓激波（Bow Shock）
    thmpb = np.linspace(0, np.pi, 101)
    x0 = 0.6
    L = 2.081
    ecc = 1.026
    rbs = L / (1 + ecc * np.cos(thmpb))
    xbs = rbs * np.cos(thmpb) + x0
    rhobs = rbs * np.sin(thmpb)

    mask = xbs <= 1.63
    xbs = xbs[mask]
    rhobs = rhobs[mask]

    theta = np.linspace(0, 2 * np.pi, 101)
    X_bs = np.tile(xbs[:, np.newaxis], (1, len(theta)))
    Y_bs = np.outer(rhobs, np.cos(theta))
    Z_bs = np.outer(rhobs, np.sin(theta))

    ax.plot_surface(X_bs, Y_bs, Z_bs, color='green', alpha=0.1, 
                    edgecolor='none', shade=True)


    # 绘制火星球体
    u, v = np.mgrid[0:2*np.pi:51j, 0:np.pi:51j]
    X_sphere = np.cos(u) * np.sin(v)
    Y_sphere = np.sin(u) * np.sin(v)
    Z_sphere = np.cos(v)
    ax.plot_surface(X_sphere, Y_sphere, Z_sphere, color='k', alpha=0.3, edgecolor='none')

    # 绘制MPB - 部分 1
    x0 = 0.640
    ecc = 0.770
    L = 1.080
    rmpb = L / (1 + ecc * np.cos(thmpb))
    xmpb = rmpb * np.cos(thmpb) + x0
    rhompb = rmpb * np.sin(thmpb)

    mask = xmpb > -0.1
    xmpb0 = xmpb[mask]
    rhompb0 = rhompb[mask]

    Y_mpb0 = np.outer(rhompb0, np.cos(theta))
    Z_mpb0 = np.outer(rhompb0, np.sin(theta))
    X_mpb0 = np.tile(xmpb0[:, np.newaxis], (1, len(theta)))

    # 注释掉的绘制指令，按需启用
    ax.plot_surface(X_mpb0, Y_mpb0, Z_mpb0, alpha=0.2, color='blue', edgecolor='none')

    # 绘制MPB - 部分 2
    x0 = 1.600
    ecc = 1.009
    L = 0.528
    rmpb = L / (1 + ecc * np.cos(thmpb))
    xmpb = rmpb * np.cos(thmpb) + x0
    rhompb = rmpb * np.sin(thmpb)

    mask = xmpb < 0.1
    xmpb1 = xmpb[mask]
    rhompb1 = rhompb[mask]

    Y_mpb1 = np.outer(rhompb1, np.cos(theta))
    Z_mpb1 = np.outer(rhompb1, np.sin(theta))
    X_mpb1 = np.tile(xmpb1[:, np.newaxis], (1, len(theta)))

    # 注释掉的绘制指令，按需启用
    ax.plot_surface(X_mpb1, Y_mpb1, Z_mpb1, alpha=0.2, color='blue', edgecolor='none')

    # 设置坐标范围和比例
    ax.set_xlim([-4, 2])
    ax.set_ylim([-7, 7])
    ax.set_zlim([-7, 7])
    ax.set_box_aspect([1, 1, 1])  # DataAspectRatio

# 使用示例
if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    BS_MPB_3d_mars(ax)
    plt.show()

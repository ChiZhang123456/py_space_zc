import numpy as np
import pyvista as pv

def generate_bs_mpb_surfaces():
    # 定义参数
    thmpb = np.linspace(0, np.pi, 101)
    theta = np.linspace(0, 2 * np.pi, 101)

    # 弓激波（Bow Shock）
    x0 = 0.6
    L = 2.081
    ecc = 1.026
    rbs = L / (1 + ecc * np.cos(thmpb))
    xbs = rbs * np.cos(thmpb) + x0
    rhobs = rbs * np.sin(thmpb)
    mask = xbs <= 1.63
    xbs = xbs[mask]
    rhobs = rhobs[mask]
    X_bs = np.tile(xbs[:, np.newaxis], (1, len(theta)))
    Y_bs = np.outer(rhobs, np.cos(theta))
    Z_bs = np.outer(rhobs, np.sin(theta))

    # 火星球体
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 51), np.linspace(0, np.pi, 51))
    X_sphere = np.cos(u) * np.sin(v)
    Y_sphere = np.sin(u) * np.sin(v)
    Z_sphere = np.cos(v)

    # MPB 部分 1
    x0 = 0.640
    ecc = 0.770
    L = 1.080
    rmpb = L / (1 + ecc * np.cos(thmpb))
    xmpb = rmpb * np.cos(thmpb) + x0
    rhompb = rmpb * np.sin(thmpb)
    mask = xmpb > -0.1
    xmpb0 = xmpb[mask]
    rhompb0 = rhompb[mask]
    X_mpb0 = np.tile(xmpb0[:, np.newaxis], (1, len(theta)))
    Y_mpb0 = np.outer(rhompb0, np.cos(theta))
    Z_mpb0 = np.outer(rhompb0, np.sin(theta))

    # MPB 部分 2
    x0 = 1.600
    ecc = 1.009
    L = 0.528
    rmpb = L / (1 + ecc * np.cos(thmpb))
    xmpb = rmpb * np.cos(thmpb) + x0
    rhompb = rmpb * np.sin(thmpb)
    mask = xmpb < 0.1
    xmpb1 = xmpb[mask]
    rhompb1 = rhompb[mask]
    X_mpb1 = np.tile(xmpb1[:, np.newaxis], (1, len(theta)))
    Y_mpb1 = np.outer(rhompb1, np.cos(theta))
    Z_mpb1 = np.outer(rhompb1, np.sin(theta))

    return {
        "bow_shock": (X_bs, Y_bs, Z_bs),
        "mars_sphere": (X_sphere, Y_sphere, Z_sphere),
        "mpb_1": (X_mpb0, Y_mpb0, Z_mpb0),
        "mpb_2": (X_mpb1, Y_mpb1, Z_mpb1),
    }

def export_to_vtk(X, Y, Z, filename):
    nx, ny = X.shape
    points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [nx, ny, 1]
    grid.save(filename)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    surfaces = generate_bs_mpb_surfaces()

    # 导出为 VTK 文件
    export_to_vtk(*surfaces["bow_shock"], "bow_shock.vtk")
    export_to_vtk(*surfaces["mars_sphere"], "mars_sphere.vtk")
    export_to_vtk(*surfaces["mpb_1"], "mpb_part1.vtk")
    export_to_vtk(*surfaces["mpb_2"], "mpb_part2.vtk")

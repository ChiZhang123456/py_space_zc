import numpy as np

def mpb_tangent_direction(Pmso, thresh=0.3):
    """
    Tailward-pointing tangent unit vectors along the empirical MPB surface.
    Only returned for points closer than `thresh` Rm to the MPB; otherwise NaN.

    Parameters
    ----------
    Pmso : (N, 3) array
        Positions in MSO coordinates [km].
    thresh : float, optional
        Distance threshold to the MPB surface [Rm]. Default 0.3.

    Returns
    -------
    t_hat : (N, 3) array
        Tailward-pointing unit tangent vectors on the MPB meridional direction.
        For points farther than `thresh` Rm from MPB, returns NaN rows.

    Notes
    -----
    MPB model in x–rho (rho = sqrt(y^2+z^2)) uses:
        R_mb = (e^2-1)(x - xF)^2 - 2 e L (x - xF) + L^2
        r_bound = sqrt(R_mb)
        dRdx = 2(e^2-1)(x - xF) - 2 e L
        dr/dx = dRdx / (2 * r_bound)   <-- 注意 1/2 因子

    3D 切向（子午线方向）构造： t ∝ e_x + (dr/dx) * e_r，其中
        e_r = (0, y/rho, z/rho) （当 rho≈0 时取 e_r=(0,1,0)）
    最后强制朝尾向（t_x < 0）。
    """

    # Mars radius [km]
    Rm = 3390.0
    eps = 1e-12

    # Normalize to Rm
    x = Pmso[:, 0] / Rm
    y = Pmso[:, 1] / Rm
    z = Pmso[:, 2] / Rm
    rho = np.sqrt(y**2 + z**2)

    # MPB parameters
    xF1, xF2 = 0.64, 1.60
    L1,  L2  = 1.08, 0.528
    e1,  e2  = 0.77, 1.009

    # Allocate
    R_mb = np.zeros_like(x)
    dRdx = np.zeros_like(x)

    # Dayside (x >= 0)
    idx_pos = x >= 0
    dx1 = x[idx_pos] - xF1
    R_mb[idx_pos]  = (e1**2 - 1.0) * dx1**2 - 2.0 * e1 * L1 * dx1 + L1**2
    dRdx[idx_pos]  = 2.0 * (e1**2 - 1.0) * dx1 - 2.0 * e1 * L1

    # Nightside (x < 0)
    idx_neg = ~idx_pos
    dx2 = x[idx_neg] - xF2
    R_mb[idx_neg]  = (e2**2 - 1.0) * dx2**2 - 2.0 * e2 * L2 * dx2 + L2**2
    dRdx[idx_neg]  = 2.0 * (e2**2 - 1.0) * dx2 - 2.0 * e2 * L2

    # Guard against tiny/negative (roundoff)
    R_mb = np.maximum(R_mb, eps)
    r_bound = np.sqrt(R_mb)

    # *** Correct slope with 1/2 factor ***
    drdx = dRdx / (2.0 * r_bound)

    # 近表面判据（与 mpb_normal 一致的思路）：
    # 用极角 theta 在该 x 上找到模型点 (x_model, rho_model)，再比较偏差
    xt = x.copy()
    xt[idx_pos] -= xF1
    xt[idx_neg] -= xF2

    ecc = np.where(idx_pos, e1, e2)
    L   = np.where(idx_pos, L1, L2)
    x0  = np.where(idx_pos, xF1, xF2)

    theta = np.arctan2(rho, xt)  # 极角
    r_model = L / (1.0 + ecc * np.cos(theta))
    x_model = r_model * np.cos(theta) + x0
    rho_model = r_model * np.sin(theta)

    close = (np.abs(x_model - x) <= thresh) & (np.abs(rho_model - rho) <= thresh)

    # 构造径向单位向量 e_r（YZ 平面）
    er_y = np.where(rho > eps, y / rho, 1.0)
    er_z = np.where(rho > eps, z / rho, 0.0)

    # Tangent vector in 3D: t ∝ e_x + (dr/dx) * e_r
    tx = np.ones_like(x)
    ty = drdx * er_y
    tz = drdx * er_z

    t = np.stack((tx, ty, tz), axis=1)

    # 归一化
    nrm = np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.clip(nrm, eps, None)
    t_hat = t / nrm

    # 强制尾向（-X_MSO）：若 x 分量 >= 0，则翻转
    flip = t_hat[:, 0] >= 0.0
    t_hat[flip] *= -1.0

    # 只保留近表面的结果，其它置 NaN
    t_hat[~close] = np.nan

    return t_hat

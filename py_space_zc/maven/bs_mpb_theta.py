import numpy as np
from scipy.interpolate import interp1d

def bs_mpb_theta(theta_deg):
    """
    Python version of the MATLAB function bs_mpb_theta.m
    Compute the Bow Shock (BS) and Magnetic Pile-Up Boundary (MPB)
    locations for a given angle theta (in degrees).

    Input:
        theta_deg : scalar or array, angle(s) in degrees from X-axis

    Output:
        res : dict with keys:
            rmpb, rbs  : radial distances to MPB and BS
            xmpb, xbs  : X-coordinates of MPB and BS
            Ryz_mpb, Ryz_bs : YZ-plane (radial) components
    """

    # ============================
    # --- Bow Shock parameters ---
    # ============================
    xbs = np.arange(1.6, -5.001, -0.001)  # same as 1.6:-0.001:-5
    x0 = 0.6
    ecc = 1.026
    L = 2.081

    Ryz_bs = np.sqrt((ecc**2 - 1) * (xbs - x0)**2 - 2 * ecc * L * (xbs - x0) + L**2)
    theta_bs = np.degrees(np.arccos(xbs / np.sqrt(xbs**2 + Ryz_bs**2)))
    Rbs = np.sqrt(xbs**2 + Ryz_bs**2)

    # ============================
    # --- MPB dayside (x > 0) ---
    # ============================
    xmpb_1 = np.arange(1.25, -0.001, -0.001)
    x0 = 0.640
    ecc = 0.770
    L = 1.080

    Ryz_mpb1 = np.sqrt((ecc**2 - 1) * (xmpb_1 - x0)**2 - 2 * ecc * L * (xmpb_1 - x0) + L**2)
    theta_mpb1 = np.degrees(np.arccos(xmpb_1 / np.sqrt(xmpb_1**2 + Ryz_mpb1**2)))

    # ============================
    # --- MPB nightside (x < 0) ---
    # ============================
    xmpb_2 = np.arange(0, -5.001, -0.001)
    x0 = 1.600
    ecc = 1.009
    L = 0.528

    Ryz_mpb2 = np.sqrt((ecc**2 - 1) * (xmpb_2 - x0)**2 - 2 * ecc * L * (xmpb_2 - x0) + L**2)
    theta_mpb2 = np.degrees(np.arccos(xmpb_2 / np.sqrt(xmpb_2**2 + Ryz_mpb2**2)))

    # combine MPB parts
    theta_mpb = np.concatenate([theta_mpb1, theta_mpb2])
    Ryz_mpb = np.concatenate([Ryz_mpb1, Ryz_mpb2])
    xmpb = np.concatenate([xmpb_1, xmpb_2])
    Rmpb = np.sqrt(xmpb**2 + Ryz_mpb**2)

    # ============================
    # --- Remove duplicate thetas ---
    # ============================
    theta_mpb, idx = np.unique(theta_mpb, return_index=True)
    Rmpb = Rmpb[idx]
    xmpb = xmpb[idx]
    Ryz_mpb = Ryz_mpb[idx]

    theta_bs, idx = np.unique(theta_bs, return_index=True)
    Rbs = Rbs[idx]
    xbs = xbs[idx]
    Ryz_bs = Ryz_bs[idx]

    # ============================
    # Interpolation (spline-like behavior)
    # ============================
    # SciPy interp1d with cubic = "spline"
    f_rmpb  = interp1d(theta_mpb, Rmpb,  kind='cubic', fill_value='extrapolate')
    f_rbs   = interp1d(theta_bs,  Rbs,   kind='cubic', fill_value='extrapolate')
    f_xmpb  = interp1d(theta_mpb, xmpb,  kind='cubic', fill_value='extrapolate')
    f_xbs   = interp1d(theta_bs,  xbs,   kind='cubic', fill_value='extrapolate')
    f_Ryz_mpb = interp1d(theta_mpb, Ryz_mpb, kind='cubic', fill_value='extrapolate')
    f_Ryz_bs  = interp1d(theta_bs,  Ryz_bs,  kind='cubic', fill_value='extrapolate')

    theta = np.asarray(theta_deg)

    res = dict(
        rmpb     = f_rmpb(theta),
        rbs      = f_rbs(theta),
        xmpb     = f_xmpb(theta),
        xbs      = f_xbs(theta),
        Ryz_mpb  = f_Ryz_mpb(theta),
        Ryz_bs   = f_Ryz_bs(theta),
    )
    return res


if __name__ == '__main__':
    import numpy as np

    theta = np.array([10, 30, 60, 90])  # degrees
    res = bs_mpb_theta(theta)

    print(res["Ryz_mpb"])
    print(res["Ryz_bs"])

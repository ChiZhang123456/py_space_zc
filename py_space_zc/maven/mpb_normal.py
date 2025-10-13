import numpy as np

def mpb_normal(P):
    """
    Compute normal vectors of the Martian Magnetic Pile-Up Boundary (MPB).

    Parameters
    ----------
    P : ndarray of shape (N, 3)
        Positions [x, y, z] in Mars radii (Rm).

    Returns
    -------
    normals : ndarray of shape (N, 3)
        Unit normal vectors at each position. If the point is too far from
        the empirical MPB surface (>0.3 Rm in x or rho), the normal is NaN.

    Notes
    -----
    MPB model parameters (Chi Zhang 2025):
      Dayside (x >= 0): ecc = 0.770, L = 1.080, x0 = 0.640
      Nightside (x < 0): ecc = 1.009, L = 0.528, x0 = 1.600

    The implicit function is used:
      F(x, rho) = (e^2 - 1)(x - x0)^2 - 2 e L (x - x0) + L^2 - rho^2 = 0

    The gradient is:
      ∂F/∂x = 2 (e^2 - 1)(x - x0) - 2 e L
      ∂F/∂rho = -2 rho

    The 3D normal is built as:
      Nx = ∂F/∂x
      Ny = ∂F/∂rho * (y/rho)
      Nz = ∂F/∂rho * (z/rho)

    Finally normalized, and forced to point sunward (+X).
    """

    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho = np.sqrt(y**2 + z**2)
    N = len(x)

    normals = np.full((N, 3), np.nan)

    # Split dayside vs nightside
    idx_dayside = x >= 0
    idx_nightside = ~idx_dayside

    # Parameters
    ecc = np.zeros(N)
    L   = np.zeros(N)
    x0  = np.zeros(N)

    ecc[idx_dayside] = 0.770
    L[idx_dayside]   = 1.080
    x0[idx_dayside]  = 0.640

    ecc[idx_nightside] = 1.009
    L[idx_nightside]   = 0.528
    x0[idx_nightside]  = 1.600

    # Model position check
    xt = x - x0
    theta = np.arctan2(rho, xt)
    r_model = L / (1 + ecc * np.cos(theta))

    x_model = r_model * np.cos(theta) + x0
    rho_model = r_model * np.sin(theta)
    rho_obs = rho

    # exclude points far from MPB (>0.3 Rm discrepancy)
    close = (np.abs(x_model - x) <= 0.3) & (np.abs(rho_model - rho_obs) <= 0.3)

    if not np.any(close):
        return normals

    # Gradients
    dFdx = 2 * (ecc**2 - 1) * xt - 2 * ecc * L
    dFdrho = -2 * rho

    # Convert to 3D components
    with np.errstate(invalid="ignore", divide="ignore"):
        drho_dy = np.where(rho > 0, y / rho, 0.0)
        drho_dz = np.where(rho > 0, z / rho, 0.0)

    Nx = dFdx
    Ny = dFdrho * drho_dy
    Nz = dFdrho * drho_dz

    normals_raw = np.stack((Nx, Ny, Nz), axis=1)

    # Normalize
    norm = np.linalg.norm(normals_raw, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    normals = normals_raw / norm

    # Enforce sunward direction (+X_MSO)
    flip = normals[:, 0] < 0
    normals[flip] *= -1.0

    # Keep NaN where point too far
    normals[~close] = np.nan

    return normals

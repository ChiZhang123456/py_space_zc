import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, TwoSlopeNorm
from typing import Optional, Sequence, Dict, Any
from .add_colorbar import add_colorbar

def plot_pcolor_quiver(
    ax: Optional[plt.Axes],
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    *,
    cmap: str = "Spectral_r",
    normalize_vectors: bool = True,
    arrow_length_factor: float = 1.0,       # <- NEW
    quiver_scale: float = 30.0,
    scale_units: str = "xy",                # <- NEW
    quiver_width: float = 0.01,
    quiver_color: str = "k",
    xscale: Optional[str] = None,
    yscale: str = "linear",
    cscale: str = "linear",
    clim: Optional[Sequence[float]] = None,
    symmetric: bool = False,
    midpoint: float = 0.0,
    linthresh: float = 1e-2,
    linscale: float = 1.0,
    colorbar: bool = True,
    cbar_label: Optional[str] = "Magnitude",
    cbar_orientation: str = "vertical",
    cbar_kwargs: Optional[Dict[str, Any]] = None,
    shading: str = "auto",
    mask_nonfinite: bool = True,
    rasterized: bool = True,
    **kwargs: Any,
):
    """
    Plot a pseudocolor mesh of vector magnitude with quiver arrows on top.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    X_grid = np.asarray(X_grid).ravel()
    Y_grid = np.asarray(Y_grid).ravel()
    U = np.asarray(U)
    V = np.asarray(V)

    # Vector magnitude
    mag = np.sqrt(U**2 + V**2)
    if mask_nonfinite:
        mag = np.ma.masked_invalid(mag)

    # Normalize arrows if requested
    if normalize_vectors:
        mag_nonzero = mag.copy()
        mag_nonzero[mag_nonzero == 0] = 1.0
        U_plot = U / mag_nonzero
        V_plot = V / mag_nonzero
    else:
        U_plot = U.copy()
        V_plot = V.copy()

    # Apply arrow length factor
    U_plot *= arrow_length_factor
    V_plot *= arrow_length_factor

    # Color normalization
    vmin, vmax = (None, None)
    if clim is not None:
        vmin, vmax = clim
    elif symmetric:
        finite = np.isfinite(mag)
        vmax = float(np.nanmax(np.abs(mag[finite]))) if np.any(finite) else 1.0
        vmin, vmax = -vmax, vmax

    if cscale == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif cscale == "log":
        mag = np.ma.masked_where(~(mag > 0), mag)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif cscale == "symlog":
        norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)
    elif cscale == "twoslope":
        if vmin is None or vmax is None:
            finite = np.isfinite(mag)
            vmin = float(np.nanmin(mag[finite]))
            vmax = float(np.nanmax(mag[finite]))
        norm = TwoSlopeNorm(vcenter=midpoint, vmin=vmin, vmax=vmax)
    else:
        raise ValueError("cscale must be one of {'linear','log','symlog','twoslope'}")

    # Axis scaling
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    # Background plot (magnitude)
    pcm = ax.pcolormesh(
        X_grid, Y_grid, mag.T, shading=shading,
        cmap=cmap, norm=norm, **kwargs
    )
    if rasterized:
        pcm.set_rasterized(True)

    # Quiver overlay
    Xg, Yg = np.meshgrid(X_grid, Y_grid)
    qs = ax.quiver(
        Xg, Yg, U_plot.T, V_plot.T,
        scale=quiver_scale,
        scale_units=scale_units,
        width=quiver_width,
        color=quiver_color,
    )

    # Colorbar
    cbar = None
    if colorbar:
        if cbar_kwargs is None:
            cbar_kwargs = {}
        cbar = add_colorbar(ax, pcm, size_ratio=0.6, **cbar_kwargs)
        if cbar_label:
            cbar.set_label(cbar_label)

    return ax, pcm, qs, cbar

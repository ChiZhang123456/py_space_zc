import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, TwoSlopeNorm
from typing import Optional, Sequence, Dict, Any
from .add_colorbar import add_colorbar

def plot_pcolor(
    ax: Optional[plt.Axes],
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    data: np.ndarray,
    *,
    xscale: Optional[str] = None,           # e.g., "linear", "log"
    yscale: str = "linear",                 # e.g., "linear", "log"
    cscale: str = "linear",                 # "linear" | "log" | "symlog" | "twoslope"
    clim: Optional[Sequence[float]] = None, # (vmin, vmax)
    cmap: Optional[str] = 'Spectral_r',             # colormap name
    symmetric: bool = False,                # auto symmetric color limits [-v, v]
    midpoint: float = 0.0,                  # center for TwoSlopeNorm
    linthresh: float = 1e-2,                # linear threshold for symlog
    linscale: float = 1.0,                  # linear scale for symlog
    colorbar: bool = True,                  # whether to add a colorbar
    cbar_label: Optional[str] = None,       # label for colorbar
    cbar_orientation: str = "vertical",     # "vertical" or "horizontal"
    cbar_kwargs: Optional[Dict[str, Any]] = None, # extra args for colorbar
    shading: str = "auto",                  # shading mode
    mask_nonfinite: bool = True,            # mask NaN/Inf
    rasterized: bool = True,                # rasterize mesh to reduce file size
    **kwargs: Any,
):
    """
    Create a robust pcolormesh plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axis to draw on. If None, create a new figure and axis.
    X_grid, Y_grid : 1D arrays
        Grid coordinates (centers). Lengths should match data shape.
    data : 2D array
        Data values on (Y, X) or (X, Y). Orientation is auto-inferred.
    xscale, yscale : {"linear", "log"} or None
        Axis scales. If None, keep current.
    cscale : {"linear", "log", "symlog", "twoslope"}
        Color normalization mode.
    clim : (vmin, vmax)
        Color limits. If None, auto from data.
    cmap : str
        Matplotlib colormap name. Default is "viridis".
    symmetric : bool
        If True and clim is None, set symmetric limits [-v, v].
    midpoint : float
        Center value for TwoSlopeNorm.
    linthresh, linscale : float
        Parameters for SymLogNorm.
    colorbar : bool
        Add a colorbar if True.
    cbar_label : str
        Label for the colorbar.
    cbar_orientation : str
        Orientation of colorbar: "vertical" or "horizontal".
    cbar_kwargs : dict
        Extra keyword args passed to fig.colorbar.
    shading : str
        Passed to pcolormesh (e.g., "auto", "nearest", "gouraud").
    mask_nonfinite : bool
        Mask NaN/Inf values. For log scale, also mask non-positive.
    rasterized : bool
        Rasterize the QuadMesh to keep vector outputs small.
    **kwargs :
        Extra arguments passed to ax.pcolormesh.

    Returns
    -------
    out : dict
        {"ax": ax, "pcm": pcm, "cbar": cbar or None}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    X_grid = np.asarray(X_grid).ravel()
    Y_grid = np.asarray(Y_grid).ravel()
    Z = np.asarray(data)

    # Auto orientation check
    if Z.shape == (len(Y_grid), len(X_grid)):
        Z_plot = Z
    elif Z.shape == (len(X_grid), len(Y_grid)):
        Z_plot = Z.T
    else:
        raise ValueError(
            f"data.shape={Z.shape} does not match (len(Y),len(X))=({len(Y_grid)},{len(X_grid)}) "
            f"or (len(X),len(Y))=({len(X_grid)},{len(Y_grid)})"
        )

    # Mask NaN/Inf
    if mask_nonfinite:
        Z_plot = np.ma.masked_invalid(Z_plot)

    # Auto symmetric color limits
    vmin, vmax = (None, None)
    if clim is not None:
        vmin, vmax = clim
    elif symmetric:
        finite = np.isfinite(Z_plot)
        v = float(np.nanmax(np.abs(Z_plot[finite]))) if np.any(finite) else 1.0
        vmin, vmax = -v, v

    # Color normalization
    if cscale == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif cscale == "log":
        if mask_nonfinite:
            Z_plot = np.ma.masked_where(~(Z_plot > 0), Z_plot)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif cscale == "symlog":
        norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)
    elif cscale == "twoslope":
        if vmin is None or vmax is None:
            finite = np.isfinite(Z_plot)
            vmin = float(np.nanmin(Z_plot[finite]))
            vmax = float(np.nanmax(Z_plot[finite]))
        norm = TwoSlopeNorm(vcenter=midpoint, vmin=vmin, vmax=vmax)
    else:
        raise ValueError("cscale must be one of {'linear','log','symlog','twoslope'}")

    # Axis scaling
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    # Plot
    pcm = ax.pcolormesh(
        X_grid, Y_grid, Z_plot, shading=shading,
        cmap=cmap or "Spectral_r", norm=norm, **kwargs
    )
    if rasterized:
        pcm.set_rasterized(True)

    # Colorbar
    cbar = None
    if colorbar:
        fig = ax.figure
        if cbar_kwargs is None:
            cbar_kwargs = {}
        cbar = add_colorbar(ax, pcm, size_ratio = 0.6, )
        if cbar_label:
            cbar.set_label(cbar_label)

    return ax, pcm, cbar




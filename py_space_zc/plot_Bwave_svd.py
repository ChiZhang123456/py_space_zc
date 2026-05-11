from py_space_zc import method, plot, ts_scalar
import matplotlib.pyplot as plt
from py_space_zc import background_B
from .gyro_information import gyro_information
import pyrfu.pyrf as pyrf
import numpy as np

def _slice_time(inp, tint):
    """Return the part of an xarray object inside tint."""
    time_dim = None
    for dim in inp.dims:
        if np.issubdtype(inp.coords[dim].dtype, np.datetime64):
            time_dim = dim
            break
    if time_dim is None:
        return inp
    return inp.sel({time_dim: slice(tint[0], tint[1])})


def _set_focus_ylim(ax, inp, margin=0.12):
    data = np.asarray(inp.data, dtype=float)
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        return

    ymin = float(np.nanmin(finite_data))
    ymax = float(np.nanmax(finite_data))
    if ymin == ymax:
        pad = abs(ymin) * margin if ymin != 0 else 1.0
    else:
        pad = (ymax - ymin) * margin
    ax.set_ylim(ymin - pad, ymax + pad)


def _focus_clim(inp, cscale="lin", percentiles=(2.0, 98.0)):
    data = np.asarray(inp.data, dtype=float)
    finite_data = data[np.isfinite(data)]
    if cscale == "log":
        finite_data = finite_data[finite_data > 0]
    if finite_data.size == 0:
        return None

    vmin, vmax = np.nanpercentile(finite_data, percentiles)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = np.nanmin(finite_data)
        vmax = np.nanmax(finite_data)
    if cscale == "log" and vmin <= 0:
        vmin = np.nanmin(finite_data[finite_data > 0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    return [float(vmin), float(vmax)]


def plot_Bwave_svd(Bwave, window_length=20.0, overlap=10.0, freq_range=[0.05, 16.0],
                   m_width_coeff=2, nav=8, planarity_min=0.6, tint_focus=None):
    """
    Perform SVD analysis on Magnetic Field data and plot wave parameters.

    Parameters:
    -----------
    Bwave : TSeries
        The input magnetic field data (typically in nT).
    window_length : float, optional
        The length of the FFT window in seconds. Default is 20.0.
    overlap : float, optional
        The overlap between windows in seconds. Default is 10.0.
    freq_range : list, optional
        The frequency range [min, max] to analyze and plot. Default is [0.05, 16.0].
    m_width_coeff : int or float, optional
        Coefficient applied to the Morlet wavelet width in `SVD_B`.
        Larger values produce a denser frequency grid, approximately
        12 * m_width_coeff bins per decade, which can help resolve harmonics.
        This improves frequency resolution at the cost of time resolution and
        computation speed. Default is 2.
    nav : int or float, optional
        Number of wave periods used for spectral-matrix averaging in `SVD_B`.
        Larger values give smoother polarization products, but reduce time
        resolution. Default is 8.
    planarity_min : float or None, optional
        If given, only keep SVD products where planarity is at least this
        value. The planarity panel itself remains unmasked. Default is 0.6.
    tint_focus : list of str or np.datetime64, optional
        The time interval for the X-axis focus (e.g., ["2025-06-16T13:05", "2025-06-16T14:00"]).
        If None, it defaults to the full range of Bwave.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
    """
    
    # --- Handle Time Interval (tint_focus) ---
    if tint_focus is None:
        # Default: use the full duration of the input Bwave
        tint_focus = [Bwave.time.data[0], Bwave.time.data[-1]]
    else:
        # Convert list of strings to numpy.datetime64 for compatibility
        tint_focus = [np.datetime64(t) for t in tint_focus]

    # --- Data Processing ---
    # Calculate background magnetic field and gyro-frequencies
    Bwave = Bwave.dropna(dim='time')
    Bwave_focus = _slice_time(Bwave, tint_focus)
    if Bwave_focus.time.size == 0:
        raise ValueError("tint_focus does not overlap the Bwave time axis")

    Bbgd = background_B(Bwave, window_length=window_length, overlap=overlap)
    Bt = pyrf.norm(Bbgd)
    # v: velocity, f: gyro-freq, r: gyro-radius
    _, f, _ = gyro_information(Bt, 100.0, 'H+') 
    fc = ts_scalar(Bt.time.data, f)     # Proton gyro-frequency
    flh = 42.9 * fc                    # Lower hybrid frequency (approx)

    # Perform Singular Value Decomposition (SVD)
    wave_res = method.SVD_B(Bwave,
                            window_length=window_length,
                            overlap=overlap,
                            freq_range=freq_range,
                            m_width_coeff=m_width_coeff,
                            nav=nav,
                            planarity_min=planarity_min)
    
    # Calculate derived parameters
    B_all = wave_res['Bperp'] + wave_res['Bpara']
    ratio = wave_res['Bpara'] / wave_res['Bperp']
    B_all_focus = _slice_time(B_all, tint_focus)
    ratio_focus = _slice_time(ratio, tint_focus)
    theta_focus = _slice_time(wave_res['theta'], tint_focus)
    ellipticity_focus = _slice_time(wave_res['ellipticity'], tint_focus)
    planarity_focus = _slice_time(wave_res['planarity'], tint_focus)

    # --- Plotting Configuration ---
    fig, axs = plot.subplot(6, 1, figsize=(10.0, 9.5), hspace=0.02, 
                            top=0.95, right=0.85, bottom=0.07, sharex=True)
    
    # Panel 0: Magnetic Field Time Series
    plot.plot_line(axs[0,0], Bwave, linewidth=1.0)
    axs[0,0].set_ylabel('B (nT)')
    axs[0,0].legend(["$B_x$", "$B_y$", "$B_z$"],
                    loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, fontsize=12, handlelength=0.5, ncol=1)
    axs[0,0].set_xlim(tint_focus)
    _set_focus_ylim(axs[0,0], Bwave_focus)

    # Panels 1-5: Spectrograms (Power, Ratio, Theta, Ellipticity, Planarity)
    _, c1 = plot.plot_spectr(axs[1,0], B_all_focus, yscale='log', cscale='log', cmap='Spectral_r',
                             clim=_focus_clim(B_all_focus, cscale="log"))
    _, c2 = plot.plot_spectr(axs[2,0], ratio_focus, yscale='log', cscale='log', cmap='coolwarm', clim=[0.1, 10.0])
    _, c3 = plot.plot_spectr(axs[3,0], theta_focus, yscale='log', cscale='lin', cmap='coolwarm', clim=[0.0, 90.0])
    _, c4 = plot.plot_spectr(axs[4,0], ellipticity_focus, yscale='log', cscale='lin', cmap='coolwarm', clim=[-1.0, 1.0])
    _, c5 = plot.plot_spectr(axs[5,0], planarity_focus, yscale='log', cscale='lin', cmap='Spectral_r', clim=[0.0, 1.0])

    # Add Panel Labels (Top right corner of each panel)
    labels = ['B Power', '$B_{para}/B_{perp}$', '$\\theta$', 'Ellipticity', 'Planarity']
    for i, txt in enumerate(labels, start=1):
        plot.add_text(axs[i,0], txt, 0.99, 0.98, color='white', facecolor='gray', fontsize=12)

    # Colorbar Adjustments
    cbars = [c1, c2, c3, c4, c5]
    c_labels = ["$B^2$ [nT$^2$/Hz]", "Ratio", "$\\theta_k$ [$^\\circ$]", "Ellipticity", "Planarity"]
    
    for i, (cb, clbl) in enumerate(zip(cbars, c_labels), start=1):
        plot.adjust_colorbar(axs[i,0], cb, pad=0.01, height_ratio=0.6, width=0.015)
        cb.set_ylabel(clbl, fontsize=10)

    # Set Global Time Title (based on original data range)
    full_tint = [Bwave.time.data[0], Bwave.time.data[-1]]
    plot.add_time_title(axs[0,0], full_tint, fontsize=14)

    # Final Axis Formatting
    plot.set_axis(axs[0,0], fontsize=12, tick_fontsize=12, label_fontsize=13)
    
    for ax in axs[1:,0]:
        # Overlay characteristic frequencies (fc and flh)
        plot.plot_line(ax, fc, color='black', linewidth=1.5)
        plot.plot_line(ax, flh, color='black', linewidth=1.5)
        
        # Apply the user-defined or default time focus
        ax.set_xlim(tint_focus)
        
        ax.set_ylabel('Freq (Hz)')
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        plot.set_axis(ax, fontsize=12, tick_fontsize=12, label_fontsize=13, 
                      ylim=(freq_range[0], freq_range[1]))

    return fig, axs


plot_Bwave_SVD = plot_Bwave_svd

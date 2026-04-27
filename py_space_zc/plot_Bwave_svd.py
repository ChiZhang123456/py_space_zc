from py_space_zc import method, plot, ts_scalar
import matplotlib.pyplot as plt
from py_space_zc import background_B
from .gyro_information import gyro_information
import pyrfu.pyrf as pyrf
import numpy as np

def plot_Bwave_svd(Bwave, window_length=20.0, overlap=10.0, freq_range=[0.05, 16.0], 
                   tint_focus=None):
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
                            freq_range=freq_range)
    
    # Calculate derived parameters
    B_all = wave_res['Bperp'] + wave_res['Bpara']
    ratio = wave_res['Bpara'] / wave_res['Bperp']

    # --- Plotting Configuration ---
    fig, axs = plot.subplot(6, 1, figsize=(10.0, 9.5), hspace=0.02, 
                            top=0.95, right=0.85, bottom=0.07, sharex=True)
    
    # Panel 0: Magnetic Field Time Series
    plot.plot_line(axs[0,0], Bwave, linewidth=1.0)
    axs[0,0].set_ylabel('B (nT)')
    axs[0,0].legend(["$B_x$", "$B_y$", "$B_z$"],
                    loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, fontsize=12, handlelength=0.5, ncol=1)

    # Panels 1-5: Spectrograms (Power, Ratio, Theta, Ellipticity, Planarity)
    _, c1 = plot.plot_spectr(axs[1,0], B_all, yscale='log', cscale='log', cmap='Spectral_r')
    _, c2 = plot.plot_spectr(axs[2,0], ratio, yscale='log', cscale='log', cmap='coolwarm', clim=[0.1, 10.0])
    _, c3 = plot.plot_spectr(axs[3,0], wave_res['theta'], yscale='log', cscale='lin', cmap='coolwarm', clim=[0.0, 90.0])
    _, c4 = plot.plot_spectr(axs[4,0], wave_res['ellipticity'], yscale='log', cscale='lin', cmap='coolwarm', clim=[-1.0, 1.0])
    _, c5 = plot.plot_spectr(axs[5,0], wave_res['planarity'], yscale='log', cscale='lin', cmap='Spectral_r', clim=[0.0, 1.0])

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
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pyrfu.pyrf import resample

def walen_test(ax, vht, ne, B, vH, tint_focus=None, 
               component='all',
               **style_kwargs):
    """
    Perform Walen Test and plot scatter with linear regression.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
    vht : array_like (1x3 vector)
    ne : TSeries (Density)
    B : TSeries (Magnetic field)
    vH : TSeries (Velocity)
    tint_focus : list/tuple, optional
    component : str, optional
        'all' (default), 'x', 'y', or 'z'.
    """

    # --- 0. Time Filtering ---
    working_vH = vH
    if tint_focus is not None:
        t_start = np.datetime64(tint_focus[0])
        t_stop = np.datetime64(tint_focus[1])
        t_mask = (vH.time.data >= t_start) & (vH.time.data <= t_stop)
        if not np.any(t_mask):
            raise ValueError("No data found within the specified range.")
        working_vH = vH[t_mask]

    # --- 1. Resampling ---
    B_res = resample(B, working_vH)
    ne_res = resample(ne, working_vH)

    # --- 2. Calculate Walen Physics ---
    v_data = np.squeeze(working_vH.data)
    dens = np.squeeze(ne_res.data)
    dens[dens <= 0] = np.nan
    
    # Va vector
    va_vec = 21.8 * B_res.data / np.sqrt(dens[:, np.newaxis])
    vht_arr = np.array(vht).flatten()
    v_res_vec = v_data - vht_arr  # V_res = V - Vht

    # --- 3. Component Selection & Logic ---
    comp_map = {'l': 0, 'm': 1, 'n': 2}
    colors_full = ['tab:red', 'tab:blue', 'tab:green']
    labels_full = ['$V_L$', '$V_M$', '$V_N$']
    
    if component.lower() == 'all':
        x_fit, y_fit = va_vec.flatten(), v_res_vec.flatten()
        plot_indices = [0, 1, 2]
        ylabel_str = r"$V - V_{HT}$ [km/s]"
    elif component.lower() in comp_map:
        idx = comp_map[component.lower()]
        x_fit, y_fit = va_vec[:, idx], v_res_vec[:, idx]
        plot_indices = [idx]
        ylabel_str = f"$V_{component} - V_{{HT,{component}}}$ [km/s]"


    # Linear Regression on selected data
    mask = ~np.isnan(x_fit) & ~np.isnan(y_fit)
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_fit[mask], y_fit[mask])

    # --- 4. Style & Axes ---
    ms = style_kwargs.get('ms', 35)
    fs_label = style_kwargs.get('fs_label', 14)
    fs_text = style_kwargs.get('fs_text', 13)
    lw = style_kwargs.get('lw', 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))

    # --- 5. Plotting ---
    for i in plot_indices:
        ax.scatter(va_vec[:, i], v_res_vec[:, i], s=ms, c=colors_full[i],
                   alpha=0.5, edgecolors='w', linewidths=0.5, label=labels_full[i])

    # Plot Fitting Line
    line_x = np.array([np.nanmin(x_fit), np.nanmax(x_fit)])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='#333333', lw=lw)

    # --- 6. Dynamic Labels ---
    # Top-right indicator
    y_pos = 0.98
    for i in plot_indices:
        ax.text(0.95, y_pos, labels_full[i], color=colors_full[i], transform=ax.transAxes, 
                fontsize=fs_text, fontweight='bold', ha='right', va='top')
        y_pos -= 0.1
    ax.text(0.95, y_pos, 'Fit', color='#333333', transform=ax.transAxes, 
            fontsize=fs_text-1, ha='right', va='top')

    # Stats Box
    res_text = f"Slope = {slope:.2f}\n$cc$ = {r_val:.2f}\nOffset = {intercept:.1f}"
    ax.text(0.05, 0.1, res_text, transform=ax.transAxes, fontsize=fs_text,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Final Adjustments
    ax.set_xlabel(r"$V_A$ [km/s]", fontsize=fs_label)
    ax.set_ylabel(ylabel_str, fontsize=fs_label)
    ax.tick_params(axis='both', labelsize=fs_label - 2)
    ax.grid(True, linestyle=':', alpha=0.5)

    return ax, {"slope": slope, "cc": r_val, "intercept": intercept}
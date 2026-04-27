import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from py_space_zc import maven, plot, norm
from matplotlib.lines import Line2D

def mvn_swia_region_id(tint):
    """
    Identifies Martian plasma regions using MAVEN SWIA and MAG data.
    
    This function implements the heuristic logic developed by Jasper Halekas
    to classify the Martian plasma environment. It uses magnetic field 
    stability, ion density, bulk energy, and spacecraft altitude.

    Parameters:
    -----------
    tint : list of str
        Time interval for data loading. 
        Format: ["YYYY-MM-DDTHH:MM", "YYYY-MM-DDTHH:MM"]
        Example: ["2021-05-08T15:00", "2021-05-08T19:30"]

    Returns:
    --------
    res : dict
        A dictionary containing:
        - 'time': numpy.datetime64 array (SWIA time resolution)
        - 'region_id': numpy.array of floats (0-5) representing the regions
        - 'labels': Dictionary mapping IDs to region names
    """

    # 1. Data Acquisition
    # Loading MAG (1Hz and High-res) and SWIA Omni-directional data
    B_high = maven.get_data(tint, 'B_high')
    B = maven.get_data(tint, 'B')
    swia = maven.get_data(tint, 'swia_omni')

    # Extract raw data components
    b_time = B["Bmso"].time.data
    b_mso = B["Bmso"].data
    p_mso = B["Pmso"].data

    swia_time = swia["N"].time.data
    swia_n = swia["N"].data.flatten()
    swia_v = swia["Vmso"].data
    swia_t = swia["Temp"].data

    fb_time = B_high["Bmso"].time.data
    fb_mso = B_high["Bmso"].data

    # 2. Pre-processing and Time Alignment
    RM = 3397.0 # Mars Radius in km

    # Convert timestamps to float64 (nanoseconds) for interpolation compatibility
    bt_n = b_time.astype('float64')
    st_n = swia_time.astype('float64')
    ft_n = fb_time.astype('float64')

    # Calculate High-frequency Magnetic Fluctuations (RMS)
    # Using a 128-point sliding window standard deviation
    b_mag_full = np.linalg.norm(fb_mso, axis=1)
    window_size = 128
    kernel = np.ones(window_size) / window_size
    # Variance formula: Var = E[X^2] - (E[X])^2
    b_std_raw = np.sqrt(np.maximum(0, np.convolve(b_mag_full**2, kernel, mode='same') - 
                                   np.convolve(b_mag_full, kernel, mode='same')**2))

    # Interpolate MAG and Position data to the SWIA time grid
    f_magstd = interp1d(ft_n, b_std_raw, bounds_error=False, fill_value="extrapolate")
    f_b = interp1d(bt_n, b_mso, axis=0, bounds_error=False, fill_value="extrapolate")
    f_p = interp1d(bt_n, p_mso, axis=0, bounds_error=False, fill_value="extrapolate")

    magstd = f_magstd(st_n)
    b_interp = f_b(st_n)
    p_interp = f_p(st_n)
    
    bx = b_interp[:, 0]
    b_mag = np.linalg.norm(b_interp, axis=1)
    px = p_interp[:, 0]
    pyz = np.linalg.norm(p_interp[:, 1:], axis=1)
    alt = np.linalg.norm(p_interp, axis=1) - RM

    # Physical conversions: Velocity (km/s) to Bulk Energy (eV)
    # Energy approx for protons: E_eV = (v_kms / 13.8)^2
    vel_mag = np.linalg.norm(swia_v, axis=1)
    E_bulk = (vel_mag / 13.8)**2
    T_avg = np.nanmean(swia_t, axis=1) if swia_t.ndim > 1 else swia_t

    # 3. Region Identification Logic
    # 0: None, 1: SW, 2: Sheath, 3: Ionos, 4: Day-Ionos, 5: Lobe
    regid = np.zeros(len(swia_time))

    # Region 1: Solar Wind
    regid[(E_bulk > 200) & (np.sqrt(T_avg)/E_bulk < 0.012) & (magstd/b_mag < 0.15) & (alt > 500)] = 1

    # Region 2: Magnetosheath
    regid[(E_bulk > 200) & ((np.sqrt(T_avg)/E_bulk > 0.015) | (magstd/b_mag > 0.25)) & (alt > 300)] = 2

    # Region 3: Ionosphere
    idx_ion = ((E_bulk < 100) | (swia_n < 0.1)) & (magstd/b_mag < 0.1) & (b_mag > 10) & (alt < 500)
    regid[idx_ion] = 3

    # Region 4: Periapsis Dayside Ionosphere (refined from Region 3)
    regid[idx_ion & (alt < 250) & (alt > 140) & ((px > 0) | (pyz > RM))] = 4

    # Region 5: Tail Lobe
    regid[(E_bulk < 200) & (magstd/b_mag < 0.1) & (np.abs(bx/b_mag) > 0.9) & (px < 0) & (alt > 300)] = 5

    # 4. Visualization
    # Mapping colors and labels for plotting
    region_colors = {
        1: '#ff4d4d',     # Solar Wind (Red)
        2: '#33cc33',     # Magnetosheath (Green)
        3: '#3399ff',     # Ionosphere (Blue)
        4: '#ffff00',     # Dayside Ionosphere (Yellow)
        5: '#9966ff'      # Tail Lobe (Purple)
    }
    labels_map = {1: 'Solar Wind', 2: 'Sheath', 3: 'Ionosphere', 4: 'Dayside Ionos.', 5: 'Tail Lobe'}

    # Configure subplots: [B-field, Density, Region Bar]
    # Height ratios make the Region Bar significantly shorter
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, 
                           gridspec_kw={'height_ratios': [1, 1, 0.25]})

    # Subplot 1: Magnetic Field Strength
    axs[0].plot(swia_time, b_mag, 'k-', lw=1)
    axs[0].set_ylabel('|B| (nT)')
    axs[0].set_title(f"MAVEN Region Identification: {tint[0][:10]}", pad=15)

    # Subplot 2: Ion Density
    axs[1].semilogy(swia_time, swia_n, 'r-', lw=1)
    axs[1].set_ylabel('Ni (cm$^{-3}$)')

    # Subplot 3: Region Shaded Bar
    # Iterating through identified segments to draw colored background spans
    for i in range(len(swia_time)-1):
        rid = regid[i]
        if rid in region_colors:
            color = region_colors[rid]
            axs[2].axvspan(swia_time[i], swia_time[i+1], color=color, alpha=0.9, lw=0)

    # Styling the Region Bar panel
    axs[2].set_yticks([])
    axs[2].set_ylabel('Region', rotation=0, labelpad=25, va='center')
    axs[2].set_xlabel('Time (UTC)')

    # Clean up grids and tick parameters
    for ax in axs:
        ax.grid(False)
        ax.tick_params(axis='both', labelsize=9)

    # Create Legend placed at the bottom center of the figure
    legend_elements = [Line2D([0], [0], color=region_colors[i], lw=6, label=labels_map[i]) 
                       for i in range(1, 6)]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
               fontsize=15, frameon=False, bbox_to_anchor=(0.5, 0.02))

    # Adjust layout to prevent overlap with the legend
    plt.tight_layout(rect=[0, 0.08, 1, 1]) 
    plt.subplots_adjust(hspace=0.12) 
    
    plt.show()

    return {"time": swia_time, "region_id": regid, "labels": labels_map}

# --- Usage Instructions ---
# Ensure 'py_space_zc' is correctly configured with your MAVEN data path.
# Call the function with a time interval list.
if __name__ == "__main__":
    # Define time range
    my_tint = ["2021-05-08T15:00", "2021-05-08T19:30"]
    
    # Run identification
    results = mvn_swia_region_id(my_tint)
    
    # Print summary
    print("Region identification complete.")
    print(f"Processed {len(results['time'])} data points.")
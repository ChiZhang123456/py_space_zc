import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Boundary Function: Bow Shock (BS) and MPB
# =========================================================
def bs_mpb(
    ax=None,
    draw_bs=True,            # Toggle Bow Shock
    draw_mpb=True,           # Toggle Magnetic Pileup Boundary
    fill_magnetosphere=False,
    sphere=True,             # Draw Mars as a solid sphere
    boundary_color='k',  # Suitable for dark backgrounds
    boundary_ls='--',        # Dashed line style
    boundary_lw=1.0,
    mars_day_color='#666666',   # Gray for dayside
    mars_night_color='#333333',  # Darker gray for nightside
    mars_color='red',        # Color if sphere=False
    mars_lw=1.5,
    mars_ls='--'
):
    """
    Plots Martian boundaries (BS and MPB) based on empirical models.
    Coordinates are typically in Martian Radii (Rm).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    th = np.linspace(0, np.pi, 201)

    # --- 1. Bow Shock (BS) Calculation ---
    if draw_bs:
        # Model parameters (Trotignon et al., 2006)
        x0_bs, L_bs, ecc_bs = 0.6, 2.081, 1.026
        rbs = L_bs / (1 + ecc_bs * np.cos(th))
        xbs = rbs * np.cos(th) + x0_bs
        rhobs = rbs * np.sin(th)
        
        # Crop to avoid extreme tail values
        mask_bs = xbs <= 1.63
        xbs, rhobs = xbs[mask_bs], rhobs[mask_bs]

        # Plot BS (North and South symmetries)
        for y in (rhobs, -rhobs):
            ax.plot(xbs, y, linestyle=boundary_ls,
                    linewidth=boundary_lw, color=boundary_color)

    # --- 2. Magnetic Pileup Boundary (MPB) Calculation ---
    if draw_mpb:
        # MPB dayside model (Vignes et al., 2000)
        x0_day, ecc_day, L_day = 0.640, 0.770, 1.080
        rmpb_day = L_day / (1 + ecc_day * np.cos(th))
        xmpb_day = rmpb_day * np.cos(th) + x0_day
        rhompb_day = rmpb_day * np.sin(th)
        mask_day = xmpb_day > 0
        xmpb0, rhompb0 = xmpb_day[mask_day], rhompb_day[mask_day]

        # MPB nightside model
        x0_night, ecc_night, L_night = 1.600, 1.009, 0.528
        rmpb_night = L_night / (1 + ecc_night * np.cos(th))
        xmpb1 = rmpb_night * np.cos(th) + x0_night
        rhompb1 = rmpb_night * np.sin(th)
        mask_night = xmpb1 < 0
        xmpb1, rhompb1 = xmpb1[mask_night], rhompb1[mask_night]

        xmpb = np.concatenate((xmpb0, xmpb1))
        rhompb = np.concatenate((rhompb0, rhompb1))

        # Optional filling of the Induced Magnetosphere
        if fill_magnetosphere:
            x_msp = np.concatenate((xmpb, xmpb[::-1]))
            y_msp = np.concatenate((rhompb, -rhompb[::-1]))
            ax.fill(x_msp, y_msp, color='lightblue', alpha=0.15, edgecolor='none')

        # Plot MPB (North and South symmetries)
        for y in (rhompb, -rhompb):
            ax.plot(xmpb, y, linestyle=boundary_ls,
                    linewidth=boundary_lw, color=boundary_color)

    # --- 3. Draw Mars Sphere ---
    if sphere:
        # Dayside semi-circle
        theta_day = np.linspace(-np.pi/2, np.pi/2, 500)
        ax.fill(np.cos(theta_day), np.sin(theta_day),
                color=mars_day_color, zorder=2)
        # Nightside semi-circle
        theta_night = np.linspace(np.pi/2, 3*np.pi/2, 500)
        ax.fill(np.cos(theta_night), np.sin(theta_night),
                color=mars_night_color, zorder=2)
    else:
        # Simple circular outline
        theta = np.linspace(0, 2*np.pi, 500)
        ax.plot(np.cos(theta), np.sin(theta), linestyle=mars_ls, 
                linewidth=mars_lw, color=mars_color)

    return ax
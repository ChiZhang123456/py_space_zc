import numpy as np
import matplotlib.pyplot as plt

def bs_mpb(ax, color=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    th = np.linspace(0, np.pi, 201)

    # === Bow Shock (BS) ===
    x0_bs, L_bs, ecc_bs = 0.6, 2.081, 1.026
    rbs = L_bs / (1 + ecc_bs * np.cos(th))
    xbs = rbs * np.cos(th) + x0_bs
    rhobs = rbs * np.sin(th)
    mask_bs = xbs <= 1.63
    xbs = xbs[mask_bs]
    rhobs = rhobs[mask_bs]

    # === MPB dayside ===
    x0_day, ecc_day, L_day = 0.640, 0.770, 1.080
    rmpb_day = L_day / (1 + ecc_day * np.cos(th))
    xmpb_day = rmpb_day * np.cos(th) + x0_day
    rhompb_day = rmpb_day * np.sin(th)
    mask_day = xmpb_day > 0
    xmpb0 = xmpb_day[mask_day]
    rhompb0 = rhompb_day[mask_day]

    # === MPB nightside ===
    x0_night, ecc_night, L_night = 1.600, 1.009, 0.528
    rmpb_night = L_night / (1 + ecc_night * np.cos(th))
    xmpb1 = rmpb_night * np.cos(th) + x0_night
    rhompb1 = rmpb_night * np.sin(th)
    mask_night = xmpb1 < 0
    xmpb1 = xmpb1[mask_night]
    rhompb1 = rhompb1[mask_night]

    # === Full MPB
    xmpb = np.concatenate((xmpb0, xmpb1))
    rhompb = np.concatenate((rhompb0, rhompb1))

    # === Fill Regions ===
    if color:
        # Magnetosheath: between BS and MPB dayside
        x_sheath = np.concatenate((xbs, xmpb0[::-1]))
        y_sheath = np.concatenate((rhobs, rhompb0[::-1]))
        #ax.fill(x_sheath, y_sheath, color='peachpuff', alpha=0.5, edgecolor='none', label='Magnetosheath')
        #ax.fill(x_sheath, -y_sheath, color='peachpuff', alpha=0.5, edgecolor='none')

        # Magnetosphere: inside MPB (day + night)
        x_msp = np.concatenate((xmpb, xmpb[::-1]))
        y_msp = np.concatenate((rhompb, -rhompb[::-1]))
        ax.fill(x_msp, y_msp, color='lightblue', alpha=0.4, edgecolor='none', label='Magnetosphere')

    # === Draw boundaries ===
    ax.plot(xbs, rhobs, 'k-', linewidth=0.6)
    ax.plot(xbs, -rhobs, 'k-', linewidth=0.6)
    ax.plot(xmpb, rhompb, 'k-', linewidth=0.6)
    ax.plot(xmpb, -rhompb, 'k-', linewidth=0.6)

    # === Draw Mars ===
    theta = np.linspace(-np.pi / 2, np.pi / 2, 500)
    ax.fill(np.cos(theta), np.sin(theta), color='white', edgecolor='none', zorder=1)
    theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 500)
    ax.fill(np.cos(theta), np.sin(theta), color='gray', edgecolor='none', zorder=1)

    # === Format axes ===
    #ax.set_xlabel(r'$X\ (R_{\rm M})$', fontsize=14)
    #ax.set_ylabel(r'$R\ (R_{\rm M})$', fontsize=14)
    ax.set_aspect('auto')
    ax.grid(False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 5)

    return ax

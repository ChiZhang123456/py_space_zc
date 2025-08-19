import matplotlib.pyplot as plt

def save_figure(fig, filename, formats=("pdf",), width_cm=16, dpi=600, bbox_inches="tight"):
    """
    Save a matplotlib figure with fixed width in cm, keeping aspect ratio.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object to save.
    filename : str
        Base filename (without extension).
    formats : str or tuple of str, optional
        Output format(s). Example: "jpg" or ("pdf", "jpg").
        Default is ("pdf",).
    width_cm : float, optional
        Desired figure width in centimeters. Default is 16 cm.
    dpi : int, optional
        Dots per inch (image resolution). Default is 600.
    bbox_inches : str, optional
        Bounding box option passed to `savefig`. Default is "tight".

    Returns
    -------
    None

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0,1], [0,1])
    >>> save_figure(fig, "figure1", formats=("pdf", "jpg"), width_cm=16)
    """
    # Convert cm to inch
    width_inch = width_cm / 2.54
    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_height / fig_width

    # Resize figure
    fig.set_size_inches(width_inch, width_inch * aspect)

    # Ensure formats is iterable
    if isinstance(formats, str):
        formats = (formats,)

    # Save in each format
    for fmt in formats:
        outname = f"{filename}.{fmt}"
        fig.savefig(outname, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {outname}")

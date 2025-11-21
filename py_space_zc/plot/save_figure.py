import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

# --- Global settings to ensure RGB output in PDF ---
mpl.rcParams['pdf.use14corefonts'] = False
mpl.rcParams['pdf.fonttype'] = 42   # TrueType -> prevents CMYK issues
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['savefig.edgecolor'] = 'white'


def save_figure(fig, filename, formats=("pdf",), dpi=600, bbox_inches="tight"):
    """
    Save a matplotlib figure with consistent RGB colors for all formats.
    Ensures PDF is saved in RGB instead of CMYK (important for Illustrator).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object to save.
    filename : str
        Base filename (without extension).
    formats : str or tuple of str, optional
        Output format(s). Example: "jpg" or ("pdf", "jpg").
    dpi : int, optional
        Output resolution (default: 600).
    bbox_inches : str, optional
        Bounding box setting ("tight" by default).
    """

    # Ensure formats is iterable
    if isinstance(formats, str):
        formats = (formats,)

    for fmt in formats:
        outname = f"{filename}.{fmt}"

        # --- Force RGB PDF (avoid CMYK issue) ---
        if fmt.lower() == "pdf":
            with PdfPages(outname) as pdf:
                pdf.savefig(fig, dpi=dpi, bbox_inches=bbox_inches)
            print(f"[RGB PDF Saved]: {outname}")

        # --- Normal image formats ---
        else:
            fig.savefig(
                outname,
                format=fmt,
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor=fig.get_facecolor(),
                edgecolor='none'
            )
            print(f"[Saved]: {outname}")

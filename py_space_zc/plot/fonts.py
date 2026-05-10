from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from pyfonts import load_google_font


FONT_FAMILY = "Roboto"
FONT_WEIGHT = "regular"


@lru_cache(maxsize=1)
def get_plot_font():
    """Return the package-wide plotting font."""
    try:
        return load_google_font(FONT_FAMILY, weight=FONT_WEIGHT)
    except Exception:
        return FontProperties(family=FONT_FAMILY, weight=FONT_WEIGHT)


def configure_plot_font(base_size: float = 12):
    """Configure Matplotlib defaults for the package-wide plotting font."""
    font = get_plot_font()
    family = font.get_name() or FONT_FAMILY

    plt.rcParams["font.size"] = base_size
    plt.rcParams["axes.titlesize"] = base_size + 2
    plt.rcParams["axes.labelsize"] = base_size + 1
    plt.rcParams["xtick.labelsize"] = base_size
    plt.rcParams["ytick.labelsize"] = base_size
    plt.rcParams["legend.fontsize"] = base_size

    plt.rcParams["font.family"] = family
    plt.rcParams["font.weight"] = FONT_WEIGHT
    plt.rcParams["axes.titleweight"] = FONT_WEIGHT
    plt.rcParams["axes.labelweight"] = FONT_WEIGHT
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.rm"] = family
    plt.rcParams["mathtext.it"] = f"{family}:italic"
    plt.rcParams["mathtext.bf"] = family
    plt.rcParams["axes.unicode_minus"] = False
    return font


def apply_plot_font(obj, font: Optional[FontProperties] = None):
    """Apply the package-wide plotting font to a figure, axis, or text object."""
    font = font or get_plot_font()

    def _apply_text(text: Text):
        size = text.get_fontsize()
        text.set_fontproperties(font)
        text.set_fontsize(size)
        text.set_fontweight(FONT_WEIGHT)

    if isinstance(obj, Text):
        _apply_text(obj)
        return obj

    fig = getattr(obj, "figure", obj)
    if hasattr(fig, "findobj"):
        for text in fig.findobj(match=Text):
            _apply_text(text)

    axes = getattr(fig, "axes", [])
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                _apply_text(text)
            title = legend.get_title()
            if title is not None:
                _apply_text(title)

    return obj

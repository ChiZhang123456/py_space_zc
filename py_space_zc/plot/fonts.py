from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from pyfonts import load_google_font


FONT_FAMILY = "Roboto"
FONT_WEIGHT = "bold"


@lru_cache(maxsize=1)
def get_plot_font():
    """Return the package-wide plotting font."""
    try:
        return load_google_font(FONT_FAMILY, weight=FONT_WEIGHT)
    except Exception:
        return FontProperties(family=FONT_FAMILY, weight=FONT_WEIGHT)


def configure_plot_font():
    """Configure Matplotlib defaults for the package-wide plotting font."""
    font = get_plot_font()
    family = font.get_name() or FONT_FAMILY
    plt.rcParams["font.family"] = family
    plt.rcParams["font.weight"] = FONT_WEIGHT
    plt.rcParams["axes.titleweight"] = FONT_WEIGHT
    plt.rcParams["axes.labelweight"] = FONT_WEIGHT
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = family
    plt.rcParams["mathtext.it"] = f"{family}:italic"
    plt.rcParams["mathtext.bf"] = f"{family}:bold"
    plt.rcParams["axes.unicode_minus"] = False
    return font


def apply_plot_font(obj, font: Optional[FontProperties] = None):
    """Apply the package-wide plotting font to a figure, axis, or text object."""
    font = font or get_plot_font()

    if isinstance(obj, Text):
        obj.set_fontproperties(font)
        obj.set_fontweight(FONT_WEIGHT)
        return obj

    fig = getattr(obj, "figure", obj)
    if hasattr(fig, "findobj"):
        for text in fig.findobj(match=Text):
            text.set_fontproperties(font)
            text.set_fontweight(FONT_WEIGHT)

    axes = getattr(fig, "axes", [])
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontproperties(font)
                text.set_fontweight(FONT_WEIGHT)
            title = legend.get_title()
            if title is not None:
                title.set_fontproperties(font)
                title.set_fontweight(FONT_WEIGHT)

    return obj

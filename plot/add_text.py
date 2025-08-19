def add_text(ax, text, x, y,
             va="top", ha="right",
             color="black", fontsize=12,
             fontweight="normal", facecolor=None,
             **kwargs):
    """
    Add a text annotation to a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to add the text to.
    text : str
        The text string to display.
    x, y : float
        Position of the text in axis coordinates (0 to 1).
    va : str, optional
        Vertical alignment. Default is 'top'.
        Options: 'top', 'bottom', 'center', 'baseline'.
    ha : str, optional
        Horizontal alignment. Default is 'right'.
        Options: 'left', 'center', 'right'.
    color : str, optional
        Text color. Default is 'black'.
    fontsize : int, optional
        Font size of the text. Default is 12.
    fontweight : str, optional
        Font weight. Default is 'normal'.
        Options: 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'.
    facecolor : str or None, optional
        Background color of the text box. Default is None (no background).
    **kwargs :
        Additional keyword arguments passed to `ax.text()`.

    Returns
    -------
    text_obj : matplotlib.text.Text
        The created text object.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> add_text(ax, "Example", 0.1, 0.9,
    ...          va="top", ha="left",
    ...          color="red", fontsize=14,
    ...          fontweight="bold", facecolor="yellow")
    >>> plt.show()
    """
    bbox = None
    if facecolor is not None:
        bbox = dict(facecolor=facecolor, edgecolor="none", boxstyle="round,pad=0.2")

    text_obj = ax.text(x, y, text,
                       transform=ax.transAxes,
                       va=va, ha=ha,
                       color=color,
                       fontsize=fontsize,
                       fontweight=fontweight,
                       bbox=bbox,
                       **kwargs)
    return text_obj

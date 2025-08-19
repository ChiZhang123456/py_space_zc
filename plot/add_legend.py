def add_legend(ax, labels=None,
               loc="center right", position=(1.01, 0.5),
               ncol=1, handlelength=2,
               frameon=False, fontsize=12, **kwargs):
    """
    Add a legend on a given axis with convenient defaults.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to attach the legend.
    labels : list of str, optional
        If provided, legend will pair these labels with ax.lines
        in drawing order (first N lines). If None, use existing labels.
    loc : str, default 'center right'
        Legend anchor location.
    position : tuple(float, float), default (1.01, 0.5)
        bbox_to_anchor for the legend.
    ncol : int, default 1
        Number of columns in legend.
    handlelength : float, default 2
        Handle length.
    frameon : bool, default False
        Whether to draw legend frame.
    fontsize : int, default 12
        Legend text size.
    **kwargs :
        Passed to ax.legend().

    Returns
    -------
    legend : matplotlib.legend.Legend
    """
    if labels is not None:
        # 用绘制顺序的线条作为 handles（不要求线自带 label）
        lines = list(ax.lines)
        if len(lines) == 0:
            # 如果没有 line，可尝试从容器中抓（bar/patch等）
            handles = ax.containers if hasattr(ax, "containers") else []
            handles = list(handles)
        else:
            handles = lines
        # 截断/对齐数量
        handles = handles[:len(labels)]
        if len(handles) < len(labels):
            # 可选：给点友好提示（不抛错，继续画）
            # print("[add_legend] Warning: fewer handles than labels; extra labels will be ignored.")
            labels = labels[:len(handles)]
        legend = ax.legend(handles, labels,
                           loc=loc, bbox_to_anchor=position,
                           ncol=ncol, handlelength=handlelength,
                           frameon=frameon, fontsize=fontsize, **kwargs)
    else:
        # 使用已有的 handle/label（要求线在 plot 时设置了 label=...）
        handles, auto_labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, auto_labels,
                           loc=loc, bbox_to_anchor=position,
                           ncol=ncol, handlelength=handlelength,
                           frameon=frameon, fontsize=fontsize, **kwargs)
    return legend

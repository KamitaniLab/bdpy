import warnings

import matplotlib.pyplot as plt
import numpy as np

from bdpy.fig import makefigure, box_off


def makeplots(
        df,
        x=None, y=None,
        x_list=None,
        subplot=None, subplot_list=None,
        figure=None, figure_list=None,
        group=None, group_list=None,
        bar_group_width=0.8,
        plot_type='bar', horizontal=False, ebar=None,
        y_lim=None, y_ticks=None,
        title=None, x_label=None, y_label=None, fontsize=12, tick_fontsize=9, points=100,
        style='default', colorset=None,
        verbose=False
    ):
    '''Make plots.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    x : str
    y : str
    x_list : list
    subplot : str
    subplot : list
    figure : str
    figure_list : list
    plot_type : {'bar', 'violin'}
    horizontal: bool
    y_lim : (y_min, y_max)
    y_ticks : array_like
    title, x_label, y_label : str
    fontsize : int
    tick_fontsize : int
    style : str
    verbose : bool

    Returns
    -------
    fig : matplotlib.figure.Figure or list of matplotlib.figure.Figure
    '''

    x_keys       = sorted(df[x].unique())
    subplot_keys = sorted(df[subplot].unique())
    figure_keys  = sorted(df[figure].unique()) if figure is not None else [None]
    group_keys   = sorted(df[group].unique()) if group is not None else [None]

    x_list       = x_keys       if x_list       is None else x_list
    subplot_list = subplot_keys if subplot_list is None else subplot_list
    figure_list  = figure_keys  if figure_list  is None else figure_list
    group_list   = group_keys   if group_list   is None else group_list

    grouping = group is not None

    if grouping:
        warnings.warn('"grouping mode" is still experimental and will not work correctly yet!')

    if verbose:
        print('X:       {}'.format(x_list))
        print('Subplot: {}'.format(subplot_list))
        if grouping:
            print('Group by: {} ({})'.format(group_keys, group_list))
        if figure is not None:
            print('Figures: {}'.format(figure_list))

    col_num = np.ceil(np.sqrt(len(subplot_list)))
    row_num = int(np.ceil(len(subplot_list) / col_num))
    col_num = int(col_num)

    figsize = (col_num * 3, row_num * 2)  # (width, height)

    if verbose:
        print('Subplot in {} x {}'.format(row_num, col_num))

    figs = []

    # Figure loop
    for fig_label in figure_list:
        if verbose:
            if fig_label is None:
                print('Creating a figure')
            else:
                print('Creating figure for {}'.format(fig_label))

        plt.style.use(style)

        #fig = makefigure('a4landscape')
        fig = plt.figure(figsize=figsize)

        # Subplot loop
        for i, sp_label in enumerate(subplot_list):
            if verbose:
                print('Creating subplot for {}'.format(sp_label))

            # Set subplot position
            col = int(i / row_num)
            row = i - col * row_num
            sbpos = (row_num - row - 1) * col_num + col + 1

            # Get data
            data = []
            for j, x_lbl in enumerate(x_list):
                if grouping:
                    data_t = []
                    for group_label in group_list:
                        if fig_label is None:
                            df_t = df.query('{} == "{}" & {} == "{}" & {} == "{}"'.format(subplot, sp_label, group, group_label, x, x_lbl))
                        else:
                            df_t = df.query('{} == "{}" & {} == "{}" & {} == "{}" & {} == "{}"'.format(subplot, sp_label, group, group_label, figure, fig_label, x, x_lbl))
                        data_tt = df_t[y].values
                        data_tt = np.array([np.nan, np.nan]) if len(data_tt) == 0 else np.concatenate(data_tt)
                        data_t.append(data_tt)
                    # violinplot requires at least two elements in the dataset
                else:
                    df_t = df.query('{} == "{}" & {} == "{}" & {} == "{}"'.format(subplot, sp_label, figure, fig_label, x, x_lbl))
                    data_t = df_t[y].values
                    data_t = np.array([np.nan, np.nan]) if len(data_t) == 0 else np.concatenate(data_t)
                    # violinplot requires at least two elements in the dataset

                data.append(data_t)

            if grouping:
                data_mean = [[np.nanmean(d) for d in data_t] for data_t in data]
            else:
                data_mean = [np.nanmean(d) for d in data]

            # Plot
            ax = plt.subplot(row_num, col_num, sbpos)

            # Lines
            if not style == 'ggplot':
                if horizontal:
                    ax.grid(axis='x', color='k', linestyle='-', linewidth=0.5)
                else:
                    ax.grid(axis='y', color='k', linestyle='-', linewidth=0.5)

            xpos = range(len(x_list))

            # Plot data
            if plot_type == 'bar':
                if grouping:
                        ydata = np.array(data_mean)
                        n_grp = ydata.shape[1]
                        w = bar_group_width / n_grp

                        for i in range(n_grp):
                            offset = i * w
                            if horizontal:
                                plt.barh(np.array(xpos) - bar_group_width / 2 + (bar_group_width / 2) * w + offset, ydata[:, i], height=w, label=group_list[i])
                            else:
                                plt.bar(np.array(xpos) - bar_group_width / 2 + (bar_group_width / 2) * w + offset, ydata[:, i], width=w, label=group_list[i])
                else:
                    if horizontal:
                        ax.barh(xpos, data_mean, color='gray')
                    else:
                        ax.bar(xpos, data_mean, color='gray')

            elif plot_type == 'violin':
                if grouping:
                    n_grp = len(data[0])
                    #w = bar_group_width / n_grp

                    for i in range(n_grp):
                        #offset = i * w
                        #x = np.array(xpos) - bar_group_width / 2 + (bar_group_width / 2) * w + offset
                        ax.violinplot(data[i], xpos, vert=not horizontal, showmeans=True, showextrema=False, showmedians=False, points=points)
                else:
                    ax.violinplot(data, xpos, vert=not horizontal, showmeans=True, showextrema=False, showmedians=False, points=points)
            else:
                raise ValueError('Unknown plot_type: {}'.format(plot_type))

            if not horizontal:
                # Vertical plot
                ax.set_xlim([-1, len(x_list)])
                ax.set_xticks(range(len(x_list)))
                if row == 0:
                    ax.set_xticklabels(x_list, rotation=-45, ha='left', fontsize=tick_fontsize)
                else:
                    ax.set_xticklabels([])

                if y_lim is None:
                    pass
                else:
                    ax.set_ylim(y_lim)
                    if y_ticks is not None:
                        ax.set_yticks(y_ticks)
                ax.tick_params(axis='y', labelsize=tick_fontsize)
            else:
                # Horizontal plot
                ax.set_ylim([-1, len(x_list)])
                ax.set_yticks(range(len(x_list)))
                if col == 0:
                    ax.set_yticklabels(x_list, fontsize=tick_fontsize)
                else:
                    ax.set_yticklabels([])

                if y_lim is None:
                    pass
                else:
                    ax.set_xlim(y_lim)
                    if y_ticks is not None:
                        ax.set_xticks(y_ticks)
                ax.tick_params(axis='x', labelsize=tick_fontsize)

            # Inset title
            x_range = plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]
            y_range = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
            tpos = (
                plt.gca().get_xlim()[0] + 0.03 * x_range,
                plt.gca().get_ylim()[1] - 0.03 * y_range
            )
            ax.text(tpos[0], tpos[1], sp_label, horizontalalignment='left', verticalalignment='top', fontsize=fontsize, bbox=dict(facecolor='white', edgecolor='none'))

            # Inset legend
            if grouping:
                plt.legend()

            box_off(ax)

            plt.tight_layout()

        # Draw X/Y labels and title ------------------------------------------
        ax = fig.add_axes([0, 0, 1, 1])
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        # X Label
        if x_label is not None:
            txt = y_label if horizontal else x_label
            ax.text(0.5, 0, txt, verticalalignment='center', horizontalalignment='center', fontsize=fontsize)

        # Y label
        if y_label is not None:
            txt = x_label if horizontal else y_label
            ax.text(0, 0.5, txt, verticalalignment='center', horizontalalignment='center', fontsize=fontsize, rotation=90)

        # Figure title
        if title is not None:
            ax.text(0.5, 0.99, '{}: {}'.format(title, fig_label), horizontalalignment='center', fontsize=fontsize)

        figs.append(fig)

    if len(figs) == 1:
        return figs[0]
    else:
        return figs

from itertools import product

from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import seaborn as sns

from bdpy.fig import box_off


def makeplots2(
        df,
        x, y,
        x_list=None,
        group=None, group_list=None,
        subplot=None, subplot_list=None,
        figure=None, figure_list=None,
        plot=None,
        horizontal=False,
        plot_size_auto=True, plot_size=(4, 0.3),
        max_col=None,
        y_lim=None, y_ticks=None,
        title=None, x_label=None, y_label=None,
        fontsize=12, tick_fontsize=9,
        style='default',
        chance_level=None, chance_level_style={'color': 'k', 'linewidth': 1},
        removenan=True,
        makeplots_kws={},
        verbose=False,
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
    plot : str | Callable
        Selectable plots can be specified as strings: 'violin', 'swarm+box', 'bar', 'line'
        You can pass your own plot function that conforms to the implementation of the `__plot_*` subfunctions.
    horizontal: bool
    plot_size : (width, height)
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
    # Check plot keys
    x_keys       = sorted(df[x].unique())
    group_keys   = sorted(df[group].unique()) if group is not None else [None]
    subplot_keys = sorted(df[subplot].unique()) if subplot is not None else [None]
    figure_keys  = sorted(df[figure].unique()) if figure is not None else [None]
    x_list       = x_keys       if x_list       is None else x_list
    group_list   = group_keys   if group_list   is None else group_list
    subplot_list = subplot_keys if subplot_list is None else subplot_list
    figure_list  = figure_keys  if figure_list  is None else figure_list
    if verbose:
        print('X:       {}'.format(x_list))
        if group is None:
            print('Group by: {} ({})'.format(group_keys, group_list))
        if subplot is not None:
            print('Subplot: {}'.format(subplot_list))
        if figure is not None:
            print('Figures: {}'.format(figure_list))

    # Check plot type
    if isinstance(plot, str):
        if plot == 'violin':
            plot = __plot_violin
        elif plot == 'swarm+box':
            plot = __plot_swarmbox
        elif plot == 'bar':
            plot = __plot_bar
        elif plot == 'line':
            plot = __plot_line
        else:
            raise ValueError('Unknown plot name: {}'.format(plot))
    elif isinstance(plot, Callable):
        pass
    else:
        raise ValueError('This type of variable cannot be specified for plot: {}'.format(plot))

    # Matrix size of plot
    col_num = np.ceil(np.sqrt(len(subplot_list)))
    row_num = int(np.ceil(len(subplot_list) / col_num))
    col_num = int(col_num)
    if max_col is not None and col_num > max_col:
        col_num = max_col
        row_num = int(np.ceil(len(subplot_list) / col_num))
    if verbose:
        print('Subplot in {} x {}'.format(row_num, col_num))

    # Plot size
    if plot_size_auto:
        if horizontal:
            plot_size = (plot_size[0], plot_size[1] * len(x_list))
        else:
            plot_size = (plot_size[0] * len(x_list), plot_size[1])

    # Figure size
    figsize = (col_num * plot_size[0], row_num * plot_size[1])  # (width, height)

    figs = []

    # Figure loop
    for fig_label in figure_list:
        if verbose:
            if fig_label is None:
                print('Creating a figure')
            else:
                print('Creating figure for {}'.format(fig_label))

        plt.style.use(style)

        fig = plt.figure(figsize=figsize)

        # Subplot loop
        for i, sp_label in enumerate(subplot_list):
            if verbose:
                print('Creating subplot for {}'.format(sp_label))

            # Set subplot position
            col = int(i / row_num)
            row = i - col * row_num
            sbpos = (row_num - row - 1) * col_num + col + 1

            # Extract data
            df_t = __strict_data(
                df,
                x, x_list,
                group, group_list,
                subplot, sp_label,
                figure, fig_label,
            )
            # Flatten data
            weird_keys = []
            for key_candidate in [figure, subplot, group, x]:
                if key_candidate is not None:
                    weird_keys.append(key_candidate)
            df_t = __weird_form_to_long(df_t, y, identify_cols=weird_keys)

            if removenan:
                df_t = df_t.dropna(subset=[y])

            # Plot
            ax = plt.subplot(row_num, col_num, sbpos)

            if not style == 'ggplot':
                if horizontal:
                    ax.grid(axis='x', color='k', linestyle='-', linewidth=0.5)
                else:
                    ax.grid(axis='y', color='k', linestyle='-', linewidth=0.5)

            legend_handler = plot(
                ax, x, y, x_list, df_t,
                horizontal=horizontal,
                group=group, group_list=group_list,
                **makeplots_kws,
            )

            ax.set(xlabel=None, ylabel=None)
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

                ax.tick_params(axis='y', labelsize=tick_fontsize, grid_color='gray',
                               grid_linestyle='--', grid_linewidth=0.8)

                if chance_level is not None:
                    plt.hlines(chance_level, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1],
                               **chance_level_style)
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

                ax.tick_params(axis='x', labelsize=tick_fontsize, grid_color='gray',
                               grid_linestyle='--', grid_linewidth=0.8)

                if chance_level is not None:
                    plt.vlines(chance_level, ymin=plt.gca().get_ylim()[0], ymax=plt.gca().get_ylim()[1],
                               **chance_level_style)

            # Inset title
            x_range = plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]
            y_range = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
            tpos = (
                plt.gca().get_xlim()[0] + 0.03 * x_range,
                plt.gca().get_ylim()[1] + 0.01 * y_range
            )
            ax.text(tpos[0], tpos[1], sp_label, horizontalalignment='left', verticalalignment='top',
                    fontsize=fontsize, bbox=dict(facecolor='white', edgecolor='none'))
            box_off(ax)
            plt.tight_layout()

        # Draw legend, X/Y labels, and title ----------------------------

        # Add legend when group is not None
        if group is not None:
            if len(subplot_list) < col_num*row_num:
                ax = plt.subplot(row_num, col_num, col_num)
            else:
                ax = fig.add_axes([1, 0.5, 1./col_num*0.6, 0.5])
            ax.legend(legend_handler[0], legend_handler[1],
                      loc='upper left', bbox_to_anchor=(0, 1.0), fontsize=tick_fontsize)
            ax.set_axis_off()

        ax = fig.add_axes([0, 0, 1, 1])
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        # X Label
        if x_label is not None:
            txt = y_label if horizontal else x_label
            ax.text(0.5, 0, txt, verticalalignment='center', horizontalalignment='center',
                    fontsize=fontsize)

        # Y label
        if y_label is not None:
            txt = x_label if horizontal else y_label
            ax.text(0, 0.5, txt, verticalalignment='center', horizontalalignment='center',
                    fontsize=fontsize, rotation=90)

        # Figure title
        if title is not None:
            if fig_label is None:
                ax.text(0.5, 0.99, title, horizontalalignment='center', fontsize=fontsize)
            else:
                ax.text(0.5, 0.99, '{}: {}'.format(title, fig_label), horizontalalignment='center', fontsize=fontsize)

        plt.tight_layout()
        figs.append(fig)

    if figure is None:
        return figs[0]
    else:
        return figs


def __plot_swarmbox(
        ax, x, y, x_list, df_t,
        horizontal=False,
        group=None, group_list=[],
        dot_color='#023eff', dot_color_palette='bright', dot_size=3, dot_alpha=0.7,
        box_color='#a1c9f4', box_color_palette='pastel', box_width=0.5, box_linewidth=0.5, box_props={'alpha': .8},
        box_meanprops=dict(linestyle='-', linewidth=1.5, color='red'),
        box_medianprops={}
):
    """
    Swarm plot & Box plot
    """
    if horizontal:
        plotx, ploty = y, x
    else:
        plotx, ploty = x, y

    # plot swarmplot
    if group is None:
        sns.swarmplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list,
            orient="h" if horizontal else "v",
            color=dot_color, size=dot_size, alpha=dot_alpha, zorder=10
        )
    else:
        sns.swarmplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list, hue=group, hue_order=group_list,
            orient="h" if horizontal else "v",
            palette=sns.color_palette(dot_color_palette, n_colors=len(group_list)),
            dodge=True,
            size=dot_size, alpha=dot_alpha, zorder=10
        )

    # plot boxplot
    if group is None:
        sns.boxplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list,
            orient="h" if horizontal else "v",
            color=box_color,
            linewidth=box_linewidth,
            width=box_width,
            showfliers=False,
            showmeans=True, meanline=True, meanprops=box_meanprops,
            medianprops=box_medianprops,
            boxprops=box_props, zorder=100
        )
        legend_handler = None
    else:
        boxax = sns.boxplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list, hue=group, hue_order=group_list,
            orient="h" if horizontal else "v",
            palette=sns.color_palette(box_color_palette, n_colors=len(group_list)),
            linewidth=box_linewidth,
            showfliers=False,
            showmeans=True, meanline=True, meanprops=box_meanprops,
            medianprops=box_medianprops,
            boxprops=box_props, zorder=100
        )
        # prepare legend
        handlers, labels = boxax.get_legend_handles_labels()
        handlers = handlers[:len(group_list)]
        labels = labels[:len(group_list)]
        if horizontal:
            legend_handler = [handlers[::-1], labels[::-1]]
        else:
            legend_handler = [handlers, labels]
        ax.get_legend().remove()

    return legend_handler


def __plot_bar(
    ax, x, y, x_list, df_t,
    horizontal=False,
    group=None, group_list=[],
    color="#023eff", color_pallete='bright', alpha=0.6,
    # bar_width=0.8, errorbar = ('ci', 95) # seaborn >= 0.12.0
):
    """
    Bar plot
    """
    if horizontal:
        plotx, ploty = y, x
    else:
        plotx, ploty = x, y

    # plot bar
    if group is None:
        barax = sns.barplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list,
            orient="h" if horizontal else "v",
            color=color, alpha=alpha
            # width=bar_width, errorbar=errorbar,
        )
        legend_handler = None
    else:
        barax = sns.barplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list, hue=group, hue_order=group_list,
            orient="h" if horizontal else "v",
            palette=sns.color_palette(color_pallete, n_colors=len(group_list)),
            alpha=alpha 
            # width=bar_width, errorbar=errorbar,
        )
        # prepare legend
        handlers, labels = barax.get_legend_handles_labels()
        handlers = handlers[:len(group_list)]
        labels = labels[:len(group_list)]
        if horizontal:
            legend_handler = [handlers[::-1], labels[::-1]]
        else:
            legend_handler = [handlers, labels]
        ax.get_legend().remove()

    return legend_handler


def __plot_violin(
    ax, x, y, x_list, df_t,
    horizontal=False,
    group=None, group_list=[],
    color='#023eff', color_palette='bright', width=0.8, alpha=0.4, points=100,
):
    '''
    Violin plot.
    * Since seaborn's violin plot does not support drawing control of mean values, 
    this is the only plot sub function based on matplotlib.
    '''
    if group is None:
        # prepare data
        xpos = np.arange(len(x_list))
        data = []
        for x_label in x_list:
            a_data = df_t.query("`{}` == '{}'".format(x, x_label))[y].values.ravel()
            if len(a_data) == 0:
                data.append([np.nan, np.nan])  # set dummy data
            else:
                data.append(a_data)
        # plot
        violinax = ax.violinplot(data, xpos, vert=not horizontal,
                                 showmeans=True, showextrema=False, showmedians=False, points=points)
        # set color to violin
        for pc in violinax['bodies']:  # body color
            pc.set_facecolor(to_rgba(color, alpha=alpha))
            pc.set_linewidth(0)
        violinax['cmeans'].set_color(to_rgba(color))  # mean color
        legend_handler = None
    else:
        n_grp = len(group_list)
        w = width / (n_grp + 1)
        xpos = np.arange(len(x_list))

        color_palette = sns.color_palette(color_palette, n_colors=n_grp).as_hex()
        legend_handlers = []
        for gi, group_label in enumerate(group_list):
            # prepare data
            offset = gi * w - (n_grp // 2) * w
            xpos_grp = np.array(xpos) + offset
            data = []
            for x_label in x_list:
                a_data = df_t.query("`{}` == '{}' and `{}` == '{}'".format(
                    x, x_label, group, group_label
                    ))[y].values.ravel()
                if len(a_data) == 0:
                    data.append([np.nan, np.nan])  # set dummy data
                else:
                    data.append(a_data)
            # plot
            violinax = ax.violinplot(
                data, xpos_grp,
                vert=not horizontal,
                showmeans=True, showextrema=False, showmedians=False, points=points,
                widths=w * 0.8)
            # set color to violin
            group_color = to_rgba(color_palette[gi], alpha=alpha)
            for pc in violinax['bodies']:  # body color
                pc.set_facecolor(group_color)
                pc.set_alpha(alpha)
                pc.set_linewidth(0)
            violinax['cmeans'].set_color(color_palette[gi])  # mean color
            # prepare legend
            legend_handlers.append(mpatches.Patch(color=group_color))

        # prepare legend
        if horizontal:
            legend_handler = [legend_handlers[::-1], group_list[::-1]]
        else:
            legend_handler = [legend_handlers, group_list]

    return legend_handler


def __plot_line(
    ax, x, y, x_list, df_t,
    horizontal=False,
    group=None, group_list=[],
    color="#4878d0", color_pallete='muted',
    errorbar=('sd', 1),
    linewidth=0.8, markerscale=0.5
):
    """
    Line plot
    """
    if horizontal:
        plotx, ploty = y, x
    else:
        plotx, ploty = x, y

    # plot bar
    if group is None:
        # plot
        lineax = sns.pointplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list,
            orient="h" if horizontal else "v",
            color=color, errorbar=errorbar,
            scale=markerscale,
        )
        # set line width
        plt.setp(lineax.lines, linewidth=linewidth)
        legend_handler = None
    else:
        # plot
        dodge = 0.4 / len(group_list)
        lineax = sns.pointplot(
            data=df_t, ax=ax,
            x=plotx, y=ploty, order=x_list, hue=group, hue_order=group_list,
            orient="h" if horizontal else "v",
            palette=sns.color_palette(color_pallete, n_colors=len(group_list)), errorbar=errorbar,
            dodge=dodge,
            scale=markerscale,
        )
        # set line width
        plt.setp(lineax.lines, linewidth=linewidth)
        # prepare legend
        handlers, labels = lineax.get_legend_handles_labels()
        handlers = handlers[:len(group_list)]
        labels = labels[:len(group_list)]
        if horizontal:
            legend_handler = [handlers[::-1], labels[::-1]]
        else:
            legend_handler = [handlers, labels]
        ax.get_legend().remove()

    return legend_handler


def __strict_data(
        df,
        x, x_list,
        group, group_list,
        subplot, sp_label,
        figure, fig_label,
):
    """
    Limit the data you need.
    Restricts records to the specified sp_label and fig_label,
    and further restricts records to the specified combination of x_list and group_list.

    Parameters
    ----------
    df: DataFrame
    x: str
    x_list: list of str
    group: str
    group_list: list of str
    subplot: str
    sp_label: str
    figure: str
    fig_label: str
    """
    # Limit by fig_label and sp_label
    base_query = []
    if fig_label is not None:
        base_query.append('`{}` == "{}"'.format(figure, fig_label))
    if sp_label is not None:
        base_query.append('`{}` == "{}"'.format(subplot, sp_label))

    # Limit by x_list and group_list
    df_list = []
    for x_label, group_label in product(x_list, group_list):
        append_query = []
        if x is not None:
            append_query.append('`{}` == "{}"'.format(x, x_label))
        if group is not None:
            append_query.append('`{}` == "{}"'.format(group, group_label))

        # Combine base & append query
        append_query.extend(base_query)
        final_query = " & ".join(append_query)
        # Extract dataframe
        df_list.append(df.query(final_query))

    # Concatenate dataframes
    df_t = pd.concat(df_list).reset_index(drop=True)

    return df_t


def __weird_form_to_long(df, target_col, identify_cols=[]):
    """
    Unpacking a DataFrame.

    Parameters
    ----------
    target_col: str
        The column name to be decomposed. Specify the column name that stores the array.
    identify_cols: list of str
        A list of columns to keep in the decomposed dataframe, other than target_col.
    """
    # # replace NaN to [np.nan]
    nan_index = np.where(df[target_col].isna())[0]
    for i in nan_index:
        df.at[i, target_col] = [np.nan]
    # Weird form
    df_result = pd.DataFrame()
    for i, row in df.iterrows():
        tmp = {}
        for col in identify_cols:
            tmp[col] = row[col]
        tmp[target_col] = row[target_col]
        df_result = pd.concat([df_result, pd.DataFrame(tmp)]).reset_index(drop=True)
    return df_result

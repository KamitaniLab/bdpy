'''Figure module

This file is a part of BdPy.

Functions
---------
makefigure
    Create a figure
box_off
    Remove upper and right axes
draw_footnote
    Draw footnote on a figure
'''


__all__ = ['makefigure', 'box_off', 'draw_footnote']



import matplotlib.pyplot as plt


def makefigure(figtype='a4landscape'):
    '''Create a figure'''

    if figtype is 'a4landscape':
        figsize = (11.7, 8.3)
    elif figtype is 'a4portrait':
        figsize = (8.3, 11.7)
    else:
        raise ValueError('Unknown figure type %s' % figtype)

    return plt.figure(figsize=figsize)


def box_off(ax):
    '''Remove upper and right axes'''
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def draw_footnote(fig, string, fontsize=9):
    '''Draw footnote on a figure'''
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.text(0.5, 0.01, string, horizontalalignment='center', fontsize=fontsize)
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()

    return ax

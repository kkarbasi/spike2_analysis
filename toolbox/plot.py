"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plot
import matplotlib.lines as lines
import matplotlib.text as text
from matplotlib import patches as patches

def pretty(fig=None, square=False, margin=0.5):
    """Beautifies a matplotlib plot

    This function takes an optional matplotlib axis object and then
    beautifies the resulting plot
    """


    font = 'sans-serif'
    fontsize = 8

    # Set the font type to obey illustrator
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['ps.useafm'] = True
    mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False  # Fixes negative sign issue (shows up as '?')

    # Get the current figure
    if fig is None:
        fig = plot.gcf()
    fig.set_facecolor("white")
    try:
        fig.tight_layout()
    except:
        pass

    # Find out how many axes there are and where they are located (e.g., rows/columns)
    x_positions = []
    y_positions = []
    for ax in fig.get_axes():
        if fig._gci() is not None and fig._gci().colorbar is not None and ax == fig._gci().colorbar.ax:
            continue # Skip color bars in calculation
        pos = ax.get_position().get_points()
        x_positions += [pos[0][0], pos[1][0]]
        y_positions += [pos[0][1], pos[1][1]]
    x_positions = np.unique(x_positions)
    y_positions = np.unique(y_positions)
    num_x_positions = len(x_positions) // 2
    num_y_positions = len(y_positions) // 2

    if square:
        image_size = np.array([1.5 * num_x_positions, 1.5 * num_y_positions])
    else:
        image_size = np.array([4 * num_x_positions, 1.5 * num_y_positions])
    fig.set_size_inches(image_size + np.array([1, 1]) * 2 * margin, forward=True)
    if num_x_positions == 1 and num_y_positions == 1 and square and len(fig.get_axes()) == 1:
        # Change the size of the axes to match typically figures (1.1")
        ax = fig.get_axes()[0]
        fig_size = (1.5 + 2 * margin)
        size_unitless = 1.1/fig_size  # With in 0,1 coordinates
        ax.set_position([(1 - size_unitless)/2, (1 - size_unitless)/2, size_unitless, size_unitless])

    for ax in fig.get_axes():
        # Get our limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if fig._gci() is not None and fig._gci().colorbar is not None and ax == fig._gci().colorbar.ax:
            continue  # Skip color bars in calculation

        # Set the background color to white
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor('white')
        else:
            ax.set_axis_bgcolor('white')

        # Find all lines and change their width to 1.5
        for o in ax.findobj(lines.Line2D):
            o.set_linewidth(1.5)

        # Turn off the box
        if ax.xaxis.get_ticks_position() == 'default':
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.tick_bottom()
        if ax.yaxis.get_ticks_position() == 'default':
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.tick_left()
        if ax.yaxis.get_ticks_position() == 'left':
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.75)
        elif ax.yaxis.get_ticks_position() == 'right':
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_linewidth(0.75)
        elif 'left' in ax.spines and 'right' in ax.spines:
            ax.spines['left'].set_linewidth(0.75)
            ax.spines['right'].set_linewidth(0.75)

        if ax.xaxis.get_ticks_position() == 'bottom':
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.75)
        elif ax.xaxis.get_ticks_position() == 'top':
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_linewidth(0.75)
        elif 'bottom' in ax.spines and 'top' in ax.spines:
            ax.spines['bottom'].set_linewidth(0.75)
            ax.spines['top'].set_linewidth(0.75)

        if 'polar' in ax.spines:
            ax.spines['polar'].set_color(None)
            ax.set_frame_on(False)
            for line in ax.get_xgridlines():
                line.set_linestyle('-')
                line.set_color('gray')
            for line in ax.get_ygridlines():
                line.set_linestyle('-')
                line.set_color('gray')

        ax.xaxis.labelpad = 2.5 # Move labels closer to the axis
        ax.yaxis.labelpad = 2.5

        # Make ticks outside the plot
        ax.tick_params(direction='out')
        ax.tick_params(length=3.5, width=0.75) # Points

        # Move the y-axis off to the left (note, we scale the response depding on the width of the figure
        movement = 0.1 / fig.get_size_inches()[0]
        if ax.yaxis.get_ticks_position() == 'left':
            ax.spines['left'].set_position(('axes', 0 - movement))
        if ax.yaxis.get_ticks_position() == 'right':
            ax.spines['right'].set_position(('axes', 1.0 + movement))
        plot.tick_params(axis='both', which='major', labelsize=8)

        # Turn off the boxes on the legend
        if ax.get_legend() is not None and len(ax.get_legend().get_texts()) > 0:
            ax.legend(frameon=False, borderaxespad=0, borderpad=0, handletextpad=0.1, handleheight=0.1)
            texts = ax.get_legend().get_texts()
            for curr_text in texts:
                curr_text.set_fontname(font)
                curr_text.set_fontsize(fontsize)
            for leg_obj in ax.get_legend().legendHandles:
                leg_obj.set_linewidth(1.5)
            for curr_line in ax.get_legend().get_lines():
                xdata = curr_line.get_xdata()
                if not len(xdata) == 2:
                    raise RuntimeError('Unexpected entry in legend')
                # Update the first point so it is half it's original length
                xdata[0] = (xdata[1] - xdata[0]) / 2 + xdata[0]
                curr_line.set_xdata(xdata)

        # Restore the axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if square:
            ax.set_aspect(1./ax.get_data_ratio())

        # Change all fonts to Arial, size 8 (after re-ticking)
        for o in ax.findobj(text.Text):
            o.set_fontname(font)
            o.set_fontsize(fontsize)

    # Update the colorbar, if there is one
    if fig._gci() is not None and fig._gci().colorbar is not None:
        c = fig._gci().colorbar
        c.ax.tick_params(direction='out')
        c.ax.tick_params(length=3.5, width=0.75)  # Points
        # For it to have the same vertical extents
        position = list(c.ax.get_position().bounds)  # List from tuple
        axis_position = ax.get_position().bounds
        position[2] *= 2 # Width
        position[1] = axis_position[1] + (axis_position[3] * 1 / 6)  # Postion of previous axis (0, 0) is bottom left
        position[3] = axis_position[3] * 2 / 3 # Height of previous axis
        c.ax.set_position(position)

        # Change all fonts to Arial, size 8 (after re-ticking)
        for o in c.ax.findobj(text.Text):
            o.set_fontname(font)
            o.set_fontsize(fontsize)

    # Draw immediately
    plot.draw()

def shaded_errorbar(x, y, err, alpha=0.25, y_error=False, **kwargs):
    """shaded_errorbar(x, y, err)

    Creates a plot with shaded errorbars (transparency) rather than
    plotting with the standard error bar technique. This is useful
    for timeseries in which there are errorbars at every point on the
    curve. The input vectors (x, y, and err) must be the same size.

    The function assumes that the error bars are symmetric amount
    the mean.


    :param x: Input vector of x values
    :param y: Input vector of y values
    :param err: Input vector of error values
    :return:
    """
    if len(x) != len(y) or len(x) != len(err):
        raise RuntimeError('Sizes of inputs are incommensurate')
    if ~isinstance(x, np.ndarray):
        x = np.array(x)
        y = np.array(y)
        err = np.array(err)

    # Plot the lines
    l = plot.plot(x, y) # Returns a tuple with length 1
    plot.setp(l, linewidth=2.0, **kwargs)

    # Plot the patch first so that the lines go on top of it
    if not y_error:
        p = patches.Polygon(np.vstack((np.concatenate((x, np.flipud(x))), np.concatenate((y + err, np.flipud(y - err))))).T,
                            alpha=alpha, facecolor=l[0].get_color(), edgecolor='none')
    else:
        p = patches.Polygon(np.vstack((np.concatenate((x + err, np.flipud(x - err))),
                                       np.concatenate((y, np.flipud(y))))).T,
                            alpha=alpha, facecolor=l[0].get_color(), edgecolor='none')
    plot.gca().add_patch(p)
    legend_on = plot.gca().get_legend() is not None
    if not legend_on:
        plot.legend(())

    current_map = plot.gca().get_legend().get_legend_handler_map()
    current_map.pop(patches.Patch, None)
    plot.gca().get_legend().update_default_handler_map(current_map)

    # Turn off the legend, if it was off before
    if not legend_on:
        plot.gca().get_legend().set_visible(False)

    return l[0]  # Return the line handle

# Test code for creating point density contours
from __future__ import division

import numpy as np

import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if __name__ == '__main__':

    # construct random arrays
    x = np.random.normal(loc=0.0, scale=0.5, size=100000)
    y = np.random.normal(loc=0.0, scale=0.5, size=100000)

    # plot arrays and contours
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y, 'o', color='midnightblue', markersize=2, markeredgecolor='None', alpha=0.9)

    # plot contours for point density
    counts, xbins, ybins = np.histogram2d(x, y, bins=25, normed=False)
    levels_to_plot = [10, 50, 200, 500, 1e3, 1.65e3]
    c = ax.contour(counts.transpose(), levels=levels_to_plot, \
        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], \
        cmap=cm.YlOrRd, linestyles='solid', interpolation='None', zorder=10)
    #ax.clabel(c, inline=True, colors='darkgoldenrod', inline_spacing=8, \
    #    fontsize=10, fontweight='black', fmt='%d', lw=2, ls='-')

    # plot colorbar inside figure
    cbaxes = inset_axes(ax, width='3%', height='50%', loc=7, bbox_to_anchor=[-0.08, 0.0, 1, 1], bbox_transform=ax.transAxes)
    cb = plt.colorbar(c, cax=cbaxes, ticks=[min(levels_to_plot), max(levels_to_plot)], orientation='vertical')
    cb.ax.get_children()[0].set_linewidths(36.0)

    # set limits and grid
    ax.grid(True)
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)

    plt.show()

    sys.exit(0)
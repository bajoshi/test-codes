import numpy as np
from astropy import units as u
from astropy.visualization import ManualInterval, ZScaleInterval, LogStretch, ImageNormalize
from astropy.visualization.wcsaxes import SphericalCircle

import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

home = os.getenv('HOME')  # does not have a trailing slash
taffy_dir = home + '/Desktop/ipac/taffy/'
taffy_extdir = home + '/Desktop/ipac/taffy_lzifu/'

sys.path.append(taffy_dir + 'codes/')
import vel_channel_map as vcm

if __name__ == '__main__':

    sdss_i, wcs_sdss = vcm.get_sdss('i')
    
    # plot
    norm = ImageNormalize(sdss_i[0].data, stretch=LogStretch())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs_sdss)
    im = ax.imshow(sdss_i[0].data, origin='lower', cmap=mpl.cm.Greys, vmin=0.05, vmax=7, norm=norm)

    ax.set_autoscale_on(False)

    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)
    lon.set_axislabel('')
    lat.set_axislabel('')

    ax.coords.frame.set_color('None')

    north_nuc = SphericalCircle((0.4245000045 * u.deg, 23.49619722 * u.deg), 0.0017047 * u.degree, \
        edgecolor='dodgerblue', facecolor='none', transform=ax.get_transform('fk5'), lw=1.5)
    south_nuc = SphericalCircle((0.4099666595 * u.deg, 23.48356417 * u.deg), 0.0017153 * u.degree, \
        edgecolor='dodgerblue', facecolor='none', transform=ax.get_transform('fk5'), lw=1.5)
    ax.add_patch(north_nuc)
    ax.add_patch(south_nuc)

    f = FontProperties()
    f.set_weight('bold')
    ax.text(0.26, 0.49, 'Bridge east', \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='green', fontproperties=f)

    f1 = f.copy()
    f1.set_weight('normal')
    ax.text(0.26, 0.6, 'Bridge east', \
        verticalalignment='top', horizontalalignment='left', \
        transform=ax.transAxes, color='green', fontproperties=f1)

    plt.show()
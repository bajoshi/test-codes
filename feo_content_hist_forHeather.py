from __future__ import division

import numpy as np
from skimage import io
import multiprocessing as mp

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter

def main():

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Read in the tif files for the FeO content 
    imarr1 = io.imread('/Users/baj/Downloads/Extract_OrientaleLP_UVVIS.tif')
    imarr2 = io.imread('/Users/baj/Downloads/Extract_GlobalLP_UVVIS.tif')
    imarr3 = io.imread('/Users/baj/Downloads/Extract_SPA_UVVIS_lp.tif')
    imarr4 = io.imread('/Users/baj/Downloads/Extract_Mare_UVVIS.tif')

    print "All arrays read in."
    print "Time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds."

    # NaN out all negative values
    #neg_idx1 = np.where((imarr1 < 0.68) & (imarr1 > 0.9))
    #imarr1[neg_idx1] == np.nan
    #print "Negative values set to NaN for image 1."
    #print "Time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds."

    #neg_idx2 = np.where((imarr2 < 0.68) & (imarr2 > 0.9))
    #imarr2[neg_idx2] == np.nan
    #print "Negative values set to NaN for image 2."
    #print "Time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds."

    #neg_idx3 = np.where((imarr3 < 0.68) & (imarr3 > 0.9))
    #imarr3[neg_idx3] == np.nan
    #print "Negative values set to NaN for image 3."
    #print "Time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds."

    #neg_idx4 = np.where((imarr4 < 0.68) & (imarr4 > 0.9))
    #imarr4[neg_idx4] == np.nan
    #print "Negative values set to NaN for image 4."
    #print "Time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds."

    # Now plot the histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # axis labels
    ax.set_xlabel(r'$\mathrm{321/415\, nm\ Ratio}$', fontsize=13)
    ax.set_ylabel(r'$\#\mathrm{pixels}$', fontsize=13)

    # actual histogram
    totalbins = 30
    (counts1, bins1, patches1) = ax.hist(imarr1.ravel(), bins=totalbins, range=(0.6, 0.9), fc='None', ec='k', histtype='step', linewidth=1.2, label='Orientale Pilot Study')
    (counts2, bins2, patches2) = ax.hist(imarr2.ravel(), bins=totalbins, range=(0.6, 0.9), fc='None', ec='r', histtype='step', linewidth=1.2, label='Global Light Plains')
    (counts3, bins3, patches3) = ax.hist(imarr3.ravel(), bins=totalbins, range=(0.6, 0.9), fc='None', ec='b', histtype='step', linewidth=1.2, label='SPA Light Plains only')
    (counts4, bins4, patches4) = ax.hist(imarr4.ravel(), bins=totalbins, range=(0.6, 0.9), fc='None', ec='g', histtype='step', linewidth=1.2, label='Mare')

    # minorticks and other stuff
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.minorticks_on()
    ax.legend(loc=1, frameon=False)
    #ax.set_yscale('log')

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,1))

    # Now add inset axes and show the two smaller histograms in there
    #ax_ins = inset_axes(ax, width="35%", height="28%", loc=7)
    #ax_ins.hist(imarr1.ravel(), bins=totalbins, range=(0.0, 40.0), fc='None', ec='k', histtype='step', linewidth=1.2)
    #ax_ins.hist(imarr3.ravel(), bins=totalbins, range=(0.0, 40.0), fc='None', ec='b', histtype='step', linewidth=1.2)
    #ax_ins.minorticks_on()

    fig.savefig('UVVIS_hist.png', dpi=300, bbox_inches='tight')

    # Now print the counts and bin edges
    for i in range(totalbins):
        print bins1[i], "    ", int(counts1[i]), "    ", int(counts2[i]), "    ", int(counts3[i]), "    ", int(counts4[i])

    print "All done."
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

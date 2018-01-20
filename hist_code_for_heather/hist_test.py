from __future__ import division

import numpy as np

import sys
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# modify rc Params
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

home = os.getenv('HOME')
testdir = home + '/Desktop/test-codes/'
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"

sys.path.append(massive_galaxies_dir + 'codes/')
import mag_hist as mh

if __name__ == '__main__':

    # read in file
    # this is a single line csv file
    orig_c = np.loadtxt(testdir + 'c12_slope.dat', delimiter=',')

    # get only finite values
    c = orig_c[np.isfinite(orig_c)]
    print "Percent of valid pixels:", len(c)*100/len(orig_c)

    # plot histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    myblue = mh.rgb_to_hex(0, 100, 180)  # define color

    totalbins = 50
    counts, bin_edges, patches = ax.hist(c, totalbins, range=[0,50], color=myblue, edgecolor='b', zorder=5)

    # print where slope value is max but also covers at least 0.5% pixels
    limiting_pix_cover = 0.005 * len(c)
    valid_count_idx = np.where(counts > limiting_pix_cover)[0]
    print "Max slope value but also covers at least 0.5% pixels:", bin_edges[valid_count_idx[-1]]
    print "Mean slope:", np.mean(c)
    #print "Median slope:", np.median(c)

    ax.axhline(y=limiting_pix_cover, ls='--', color='r', zorder=10)

    ax.set_xticklabels(ax.get_xticks().tolist(), size=12)
    ax.set_yticklabels(ax.get_yticks().tolist(), size=12)

    ax.minorticks_on()
    ax.tick_params('both', width=1, length=3, which='minor')
    ax.tick_params('both', width=1, length=4.7, which='major')
    ax.grid(True, alpha=0.4)

    #plt.show()
    
    sys.exit(0)
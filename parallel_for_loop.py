"""
This code tests running a for loop on multiple cores i.e. in parallel.
It uses the model modifications done to the BC03 models, while trying to
find a grism-z, as a template function which needs to be parallelized.

See the code new_refine_grismz.py for the original for loop.
"""

from __future__ import division

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
import multiprocessing

import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

def nsum(n, const):
    nsum = 0
    
    for i in range(1, n+1):
        nsum = nsum + i + const

    return nsum

def do_basic_test(num_cores):
    """
    One caveat: The computation done within the function to be parallelized has to 
    be intensive enough that it justifies parallelization. 
    """

    total_num = int(1e4)
    constant = 10

    #### ----------------------------- Parallel for loop ----------------------------- ####
    # Start time
    start = time.time()
    a = Parallel(n_jobs=num_cores) (delayed(nsum)(i, constant) for i in range(total_num))

    # total time up to now
    print "Total time taken up to now --", time.time() - start, "seconds. With", num_cores, "CPUs."

    #### ----------------------------- Classical for loop ---------------------------- ####
    time_after_par_forloop = time.time()
    res = []
    for j in range(total_num):
        res.append(nsum(j, constant))

    # total time for this for loop
    print "Total time taken up to now --", time.time() - time_after_par_forloop, "seconds. One one CPU."

    return None

if __name__ == '__main__':

    do_basic_test(3)

    sys.exit(0)
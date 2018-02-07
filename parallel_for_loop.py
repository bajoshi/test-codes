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

def sqr(i):
    return i*i

def do_model_modifications(model_index_number, lsf, resampling_lam_grid, model_comp_spec, model_lam_grid_z):

    return current_modified_model

def call_func():

    #bc03_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')

    return result

def do_basic_test():

    total_num = int(1e4)
    num_cores = multiprocessing.cpu_count()

    # Start time
    start = time.time()

    #### ----------------------------- Parallel for loop ----------------------------- ####
    a = Parallel(n_jobs=1)(delayed(sqr)(i) for i in range(total_num))

    # total time up to now
    time_after_par_forloop = time.time()
    print "Total time taken up to now --", time.time() - start, "seconds."

    #### ----------------------------- Classical for loop ----------------------------- ####
    res = []
    for j in range(total_num):
        res.append(sqr(j))

    # total time for this for loop
    print "Total time taken up to now --", time.time() - time_after_par_forloop, "seconds."

    # ----------------------------- check if arrays are equal ----------------------------- # 
    a = np.asarray(a)
    res = np.asarray(res)

    if np.array_equal(a, res):
        print True

    return None

if __name__ == '__main__':

    do_basic_test()

    sys.exit(0)
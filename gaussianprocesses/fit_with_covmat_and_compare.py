from __future__ import division

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata
from scipy.integrate import simps

import os
import sys
import time
import datetime

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"
threedhst_datadir = home + "/Desktop/3dhst_data/"
massive_figures_dir = figs_dir + 'massive-galaxies-figures/'
savedir_photoz = massive_figures_dir + 'photoz_run_jan2019/'  # Required to save p(z) curve and z_arr
savedir_spz = massive_figures_dir + 'spz_run_jan2019/'  # Required to save p(z) curve and z_arr
savedir_grismz = massive_figures_dir + 'grismz_run_jan2019/'  # Required to save p(z) curve and z_arr

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
sys.path.append(home + '/Desktop/test-codes/cython_test/cython_profiling/')
sys.path.append(home + '/Desktop/test-codes/gaussianprocesses/')
import refine_redshifts_dn4000 as old_ref
from fullfitting_grism_broadband_emlines import do_fitting, get_flam, get_flam_nonhst
from photoz import do_photoz_fitting_lookup
from new_refine_grismz_gridsearch_parallel import get_data
import model_mods as mm
from covmat_test import get_covmat

speed_of_light = 299792458e10  # angstroms per second

def main():

    # ------------------------------ Get models ------------------------------ #
    # read in entire model set
    # To see how these arrays were created check the code:
    # $HOME/Desktop/test-codes/shared_memory_multiprocessing/shmem_parallel_proc.py
    # This part will fail if the arrays dont already exist.
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)

    log_age_arr = np.load(figs_data_dir + 'log_age_arr.npy', mmap_mode='r')
    metal_arr = np.load(figs_data_dir + 'metal_arr.npy', mmap_mode='r')
    nlyc_arr = np.load(figs_data_dir + 'nlyc_arr.npy', mmap_mode='r')
    tau_gyr_arr = np.load(figs_data_dir + 'tau_gyr_arr.npy', mmap_mode='r')
    tauv_arr = np.load(figs_data_dir + 'tauv_arr.npy', mmap_mode='r')
    ub_col_arr = np.load(figs_data_dir + 'ub_col_arr.npy', mmap_mode='r')
    bv_col_arr = np.load(figs_data_dir + 'bv_col_arr.npy', mmap_mode='r')
    vj_col_arr = np.load(figs_data_dir + 'vj_col_arr.npy', mmap_mode='r')
    ms_arr = np.load(figs_data_dir + 'ms_arr.npy', mmap_mode='r')
    mgal_arr = np.load(figs_data_dir + 'mgal_arr.npy', mmap_mode='r')

    model_lam_grid_withlines_mmap = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines_mmap = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", time.time() - start, "seconds."

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)

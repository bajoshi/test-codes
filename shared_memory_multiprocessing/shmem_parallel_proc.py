from __future__ import division

import numpy as np
from astropy.io import fits
import multiprocessing as mp

import os
import sys

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(massive_galaxies_dir + 'codes/')
sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import refine_redshifts_dn4000 as old_ref
from fullfitting_grism_broadband_emlines import do_fitting, get_flam, get_flam_nonhst
from photoz import do_photoz_fitting_lookup
from new_refine_grismz_gridsearch_parallel import get_data

def do_work(model_comp_spec_withlines, test_arr, current_id, current_field):
    # Some dummy computations using the shared array

    avg = np.mean(model_comp_spec_withlines)

    print "Average:", avg

    med = np.median(model_comp_spec_withlines)

    print "Median:", med

    res = (model_comp_spec_withlines - test_arr) / test_arr

    print "Residuals with a test array where all elements=1e-3"
    print "Residual shape:", res.shape
    print "Residual:", res

    return None

if __name__ == '__main__':

    # ------------------------------- Get correct directories ------------------------------- #
    figs_data_dir = '/Volumes/Bhavins_backup/bc03_models_npy_spectra/'
    threedhst_datadir = "/Volumes/Bhavins_backup/3dhst_data/"
    cspout = "/Volumes/Bhavins_backup/bc03_models_npy_spectra/cspout_2016updated_galaxev/"
    # This is if working on the laptop. 
    # Then you must be using the external hard drive where the models are saved.
    if not os.path.isdir(figs_data_dir):
        figs_data_dir = figs_dir  # this path only exists on firstlight
        threedhst_datadir = home + "/Desktop/3dhst_data/"  # this path only exists on firstlight
        cspout = home + '/Documents/galaxev_bc03_2016update/bc03/src/cspout_2016updated_galaxev/'
        if not os.path.isdir(figs_data_dir):
            print "Model files not found. Exiting..."
            sys.exit(0)

    """
    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # ------------------------------ Get models ------------------------------ #
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_data_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)
    model_lam_grid_withlines = np.load(figs_data_dir + 'model_lam_grid_withlines.npy')
    model_comp_spec_withlines = np.load(figs_data_dir + 'model_comp_spec_withlines.npy')

    # total run time up to now
    print "All models now in numpy array and have emission lines. Total time taken up to now --", 
    print time.time() - start, "seconds."

    # ---------------------------------- Read in look-up tables for model mags ------------------------------------- #
    # Using the look-up table now since it should be much faster
    # First get them all into an appropriate shape
    u = np.load(figs_data_dir + 'all_model_mags_par_u.npy')
    f435w = np.load(figs_data_dir + 'all_model_mags_par_f435w.npy')
    f606w = np.load(figs_data_dir + 'all_model_mags_par_f606w.npy')
    f775w = np.load(figs_data_dir + 'all_model_mags_par_f775w.npy')
    f850lp = np.load(figs_data_dir + 'all_model_mags_par_f850lp.npy')
    f125w = np.load(figs_data_dir + 'all_model_mags_par_f125w.npy')
    f140w = np.load(figs_data_dir + 'all_model_mags_par_f140w.npy')
    f160w = np.load(figs_data_dir + 'all_model_mags_par_f160w.npy')
    irac1 = np.load(figs_data_dir + 'all_model_mags_par_irac1.npy')
    irac2 = np.load(figs_data_dir + 'all_model_mags_par_irac2.npy')
    irac3 = np.load(figs_data_dir + 'all_model_mags_par_irac3.npy')
    irac4 = np.load(figs_data_dir + 'all_model_mags_par_irac4.npy')

    # put them in a list since I need to iterate over it
    all_model_flam = [u, f435w, f606w, f775w, f850lp, f125w, f140w, f160w, irac1, irac2, irac3, irac4]

    # cnovert to numpy array
    all_model_flam = np.asarray(all_model_flam)

    # ------------------------------- Read in photometry catalogs ------------------------------- #
    # GOODS-N from 3DHST
    # The photometry and photometric redshifts are given in v4.1 (Skelton et al. 2014)
    # The combined grism+photometry fits, redshifts, and derived parameters are given in v4.1.5 (Momcheva et al. 2016)
    photometry_names = ['id', 'ra', 'dec', 'f_F160W', 'e_F160W', 'f_F435W', 'e_F435W', 'f_F606W', 'e_F606W', \
    'f_F775W', 'e_F775W', 'f_F850LP', 'e_F850LP', 'f_F125W', 'e_F125W', 'f_F140W', 'e_F140W', \
    'f_U', 'e_U', 'f_IRAC1', 'e_IRAC1', 'f_IRAC2', 'e_IRAC2', 'f_IRAC3', 'e_IRAC3', 'f_IRAC4', 'e_IRAC4', \
    'IRAC1_contam', 'IRAC2_contam', 'IRAC3_contam', 'IRAC4_contam']
    goodsn_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 15,16, 27,28, 39,40, 45,46, 48,49, 54,55, 12,13, 63,64, 66,67, 69,70, 72,73, 90,91,92,93), skip_header=3)
    goodss_phot_cat_3dhst = np.genfromtxt(threedhst_datadir + 'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat', \
        dtype=None, names=photometry_names, \
        usecols=(0,3,4, 9,10, 18,19, 30,31, 39,40, 48,49, 54,55, 63,64, 15,16, 75,76, 78,79, 81,82, 84,85, 130,131,132,133), skip_header=3)

    # Read in Vega spectrum and get it in the appropriate forms
    vega = np.genfromtxt(massive_galaxies_dir + 'grismz_pipeline/' + 'vega_reference.dat', dtype=None, \
        names=['wav', 'flam'], skip_header=7)

    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # ------------------------------ Test parallel processing ------------------------------ #
    # I'm testing my parallel processing with shared memory mapped arrays by running
    # dummy calcuations that simulate the actual SPZ code.
    """
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)
    lamsize = 13238
    model_lam_grid_withlines = np.load(figs_data_dir + 'model_lam_grid_withlines.npy', mmap_mode='r')
    model_comp_spec_withlines = np.load(figs_data_dir + 'model_comp_spec_withlines.npy', mmap_mode='r')
    # If using np.memmap, shape has to be specified otherwise np.memmap will return a 1D array by default
    # dtype also has to be specified because default is uint8

    """
    print type(model_comp_spec_withlines)
    print model_comp_spec_withlines.shape
    print model_comp_spec_withlines.dtype
    print model_comp_spec_withlines.size
    print sys.getsizeof(model_comp_spec_withlines), model_comp_spec_withlines.size * 8
    print model_comp_spec_withlines
    """

    test_arr = np.ones(model_comp_spec_withlines.shape) * 1e-3
    test_arr = np.save(figs_data_dir + 'test_arr_for_mmap.npy', test_arr)
    del test_arr
    print "Test array for memory map saved. Object deleted."

    test_arr_mmap_read = np.load(figs_data_dir + 'test_arr_for_mmap.npy', mmap_mode='r')

    current_id = 12345
    current_field = 'GOODS'

    processes = [mp.Process(target=do_work, args=(model_comp_spec_withlines, test_arr_mmap_read, current_id, current_field)) for i in xrange(2)]
    for p in processes:
        p.start()
        print "Current process ID:", p.pid
    for p in processes:
        p.join()

    sys.exit(0)
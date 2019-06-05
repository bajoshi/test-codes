from __future__ import division

import numpy as np
from astropy.io import fits
import multiprocessing as mp

import os
import sys
import time
import datetime

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
from run_final_sample import get_all_redshifts

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

def try_small_test():

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

    if not os.path.isfile(figs_data_dir + 'test_arr_for_mmap.npy'):
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

    return None

def do_hdr2npy(figs_data_dir):

    # Read in HDUList for header info corresponding to spectra
    bc03_all_spec_hdulist = fits.open(figs_data_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')

    # Now save header info for each HDU into numpy arrays 
    log_age_arr = np.zeros(total_models)
    metal_arr = np.zeros(total_models)
    nlyc_arr = np.zeros(total_models)
    tau_gyr_arr = np.zeros(total_models)
    tauv_arr = np.zeros(total_models)
    ub_col_arr = np.zeros(total_models)
    bv_col_arr = np.zeros(total_models)
    vj_col_arr = np.zeros(total_models)
    ms_arr = np.zeros(total_models)
    mgal_arr = np.zeros(total_models)
    for k in range(total_models):
        log_age_arr[k] = float(bc03_all_spec_hdulist[k+1].header['LOGAGE'])
        metal_arr[k] = float(bc03_all_spec_hdulist[k+1].header['METAL'])
        nlyc_arr[k] = float(bc03_all_spec_hdulist[k+1].header['NLYC'])
        tau_gyr_arr[k] = float(bc03_all_spec_hdulist[k+1].header['TAUGYR'])
        tauv_arr[k] = float(bc03_all_spec_hdulist[k+1].header['TAUV'])
        ub_col_arr[k] = float(bc03_all_spec_hdulist[k+1].header['UBCOL'])
        bv_col_arr[k] = float(bc03_all_spec_hdulist[k+1].header['BVCOL'])
        vj_col_arr[k] = float(bc03_all_spec_hdulist[k+1].header['VJCOL'])
        ms_arr[k] = float(bc03_all_spec_hdulist[k+1].header['MS'])
        mgal_arr[k] = float(bc03_all_spec_hdulist[k+1].header['MGAL'])

    np.save(figs_data_dir + 'log_age_arr.npy', log_age_arr)
    np.save(figs_data_dir + 'metal_arr.npy', metal_arr)
    np.save(figs_data_dir + 'nlyc_arr.npy', nlyc_arr)
    np.save(figs_data_dir + 'tau_gyr_arr.npy', tau_gyr_arr)
    np.save(figs_data_dir + 'tauv_arr.npy', tauv_arr)
    np.save(figs_data_dir + 'ub_col_arr.npy', ub_col_arr)
    np.save(figs_data_dir + 'bv_col_arr.npy', bv_col_arr)
    np.save(figs_data_dir + 'vj_col_arr.npy', vj_col_arr)
    np.save(figs_data_dir + 'ms_arr.npy', ms_arr)
    np.save(figs_data_dir + 'mgal_arr.npy', mgal_arr)

    bc03_all_spec_hdulist.close()
    del bc03_all_spec_hdulist

    return None

if __name__ == '__main__':

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

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

    # Try running this "small test" first
    # try_small_test()

    # ------------------------------- Get catalog for final sample ------------------------------- #
    final_sample = np.genfromtxt(massive_galaxies_dir + 'spz_paper_sample.txt', dtype=None, names=True)

    # ------------------------------ Get models ------------------------------ #
    # read in entire model set
    total_models = 37761 # get_total_extensions(bc03_all_spec_hdulist)

    # COnvert header info to numpy arrays
    if not os.path.isfile(figs_data_dir + 'log_age_arr.npy'):
        # only checking for one of these arrays that I need. 
        # Assuming that if one is missing then all probably are
        do_hdr2npy(figs_data_dir)

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
    print "All models now in numpy array and have emission lines. Total time taken up to now --", 
    print time.time() - start, "seconds."

    # ---------------------------------- Read in look-up tables for model mags ------------------------------------- #
    # Using the look-up table now since it should be much faster
    # First get them all into an appropriate shape
    if not os.path.isfile(figs_data_dir + 'all_model_flam.npy'):
        u_mmap = np.load(figs_data_dir + 'all_model_mags_par_u.npy', mmap_mode='r')
        f435w_mmap = np.load(figs_data_dir + 'all_model_mags_par_f435w.npy', mmap_mode='r')
        f606w_mmap = np.load(figs_data_dir + 'all_model_mags_par_f606w.npy', mmap_mode='r')
        f775w_mmap = np.load(figs_data_dir + 'all_model_mags_par_f775w.npy', mmap_mode='r')
        f850lp_mmap = np.load(figs_data_dir + 'all_model_mags_par_f850lp.npy', mmap_mode='r')
        f125w_mmap = np.load(figs_data_dir + 'all_model_mags_par_f125w.npy', mmap_mode='r')
        f140w_mmap = np.load(figs_data_dir + 'all_model_mags_par_f140w.npy', mmap_mode='r')
        f160w_mmap = np.load(figs_data_dir + 'all_model_mags_par_f160w.npy', mmap_mode='r')
        irac1_mmap = np.load(figs_data_dir + 'all_model_mags_par_irac1.npy', mmap_mode='r')
        irac2_mmap = np.load(figs_data_dir + 'all_model_mags_par_irac2.npy', mmap_mode='r')
        irac3_mmap = np.load(figs_data_dir + 'all_model_mags_par_irac3.npy', mmap_mode='r')
        irac4_mmap = np.load(figs_data_dir + 'all_model_mags_par_irac4.npy', mmap_mode='r')

        # put them in a list since I need to iterate over it
        all_model_flam = [u_mmap, f435w_mmap, f606w_mmap, f775w_mmap, f850lp_mmap, f125w_mmap, f140w_mmap, f160w_mmap, irac1_mmap, irac2_mmap, irac3_mmap, irac4_mmap]

        # cnovert to numpy array and save
        all_model_flam = np.array(all_model_flam)
        np.save(figs_data_dir + 'all_model_flam.npy', all_model_flam)

    all_model_flam_mmap = np.load(figs_data_dir + 'all_model_flam.npy', mmap_mode='r')

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

    speed_of_light = 299792458e10  # angstroms per second
    vega_lam = vega['wav']
    vega_spec_flam = vega['flam']
    vega_nu = speed_of_light / vega_lam
    vega_spec_fnu = vega_lam**2 * vega_spec_flam / speed_of_light

    # ------------------------------ Test parallel processing ------------------------------ #
    print "Starting parallel processing. Will run each galaxy on a separate core."
    print "Total time taken up to now --", str("{:.2f}".format(time.time() - start)), "seconds."
    total_final_sample = len(final_sample)
    galaxy_count = 0
    num_cores = 4

    processes = [mp.Process(target=get_all_redshifts, args=(final_sample['pearsid'][j], final_sample['field'][j], final_sample['ra'][j], final_sample['dec'][j], 
        final_sample['zspec'][j], goodsn_phot_cat_3dhst, goodss_phot_cat_3dhst, vega_spec_fnu, vega_spec_flam, vega_nu, vega_lam, \
        model_lam_grid_withlines_mmap, model_comp_spec_withlines_mmap, all_model_flam_mmap, total_models, start, \
        log_age_arr, metal_arr, nlyc_arr, tau_gyr_arr, tauv_arr, ub_col_arr, bv_col_arr, vj_col_arr, ms_arr, mgal_arr)) for j in xrange(num_cores)]
    for p in processes:
        p.start()
        print "Current process ID:", p.pid
    for p in processes:
        p.join()

    print "Finished", num_cores, "galaxies on", num_cores, "cores."
    print "Done with shared memory and parallel testing. Exiting."
    print "Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."

    """
    zp_minchi2, zp, zp_zerr_low, zp_zerr_up, zp_min_chi2, zp_bestalpha, zp_model_idx, zp_age, zp_tau, zp_av, \
    zspz_minchi2, zspz, zspz_zerr_low, zspz_zerr_up, zspz_min_chi2, zspz_bestalpha, zspz_model_idx, zspz_age, zspz_tau, zspz_av, \
    zg_minchi2, zg, zg_zerr_low, zg_zerr_up, zg_min_chi2, zg_bestalpha, zg_model_idx, zg_age, zg_tau, zg_av
    """

    sys.exit(0)
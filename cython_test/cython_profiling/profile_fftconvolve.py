from __future__ import division

import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

sys.path.append(figs_dir + 'stacking-analysis-pears/codes/')
import grid_coadd as gd

import simple_fftconvolve_test

if __name__ == '__main__':

    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542
    
    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)
    
    model_comp_spec = np.zeros((total_models, len(model_lam_grid)), dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data
    
    print "All models read."

    # Read in LSF
    current_id = 91095
    current_field = 'GOODS-S'
    current_photoz = 0.97
    
    lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(current_id, current_photoz, current_field)

    lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
    lsf = np.genfromtxt(lsf_filename)

    # Now check the FFT convolution of these models your way and the Scipy way
    # They should be the same
    for k in range(total_models):

        conv_model = simple_fftconvolve_test.simple_1d_fftconvolve(model_comp_spec[k], lsf)
        conv_model_scipy = fftconvolve(model_comp_spec[k], lsf)

        if not np.allclose(conv_model, conv_model_scipy):
            print "At model:", i
            print "The two convolutions do not give the same answer. Exiting."
            sys.exit(0)

    print "Did not break inside loop. All done."
    sys.exit(0)
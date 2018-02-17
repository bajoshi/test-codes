#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

#import pyximport
#pyximport.install()

import numpy as np

import model_mods_cython_copytoedit

# Fake data
"""
model_lam_grid = np.arange(2000., 7000., 1.)
total_models = 34542
model_comp_spec = np.ones((total_models, len(model_lam_grid)), dtype=np.float64)
resampling_lam_grid = np.arange(5200., 10400., 40.)
lsf = np.array([0,0,0,0,0,0,0.25,0.45,0.6,0.8,1.0,0.8,0.6,0.45,0.25,0,0,0,0,0,0], dtype=np.float64)
z = 0.98
"""

# real galaxy data
# Check using this once you've confirmed 
# that its working with the fake data
import sys
import os
from astropy.io import fits
home = os.getenv('HOME')
figs_dir = home + '/Desktop/FIGS/'
sys.path.append(figs_dir + 'stacking-analysis-pears/codes/')
sys.path.append(figs_dir + 'massive-galaxies/codes/')
import grid_coadd as gd
import refine_redshifts_dn4000 as old_ref

if __name__ == '__main__':
    
    obj_id = 91095
    obj_field = 'GOODS-S'
    obj_photoz = 0.97
    
    lam_em, flam_em, ferr_em, specname, pa_chosen, netsig_chosen = gd.fileprep(obj_id, obj_photoz, obj_field)
    
    print "Netsig:", netsig_chosen
    
    flam_obs = flam_em / (1 + obj_photoz)
    ferr_obs = ferr_em / (1 + obj_photoz)
    lam_obs = lam_em * (1 + obj_photoz)
    
    lsf_filename = home + "/Desktop/FIGS/new_codes/pears_lsfs/south_lsfs/" + "s" + \
    str(obj_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
    lsf = np.loadtxt(lsf_filename)
    
    # extend lam_grid to be able to move the lam_grid later 
    avg_dlam = old_ref.get_avg_dlam(lam_obs)
    
    lam_low_to_insert = np.arange(5000, lam_obs[0], avg_dlam, dtype=np.float64)
    lam_high_to_append = np.arange(lam_obs[-1] + avg_dlam, 10500, avg_dlam, dtype=np.float64)
    
    resampling_lam_grid = np.insert(lam_obs, obj=0, values=lam_low_to_insert)
    resampling_lam_grid = np.append(resampling_lam_grid, lam_high_to_append)
    
    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542
    
    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    #print model_lam_grid.dtype
    #model_lam_grid = model_lam_grid.byteswap().newbyteorder()
    #print model_lam_grid.dtype
    model_lam_grid = model_lam_grid.astype(np.float64)
    #print model_lam_grid.dtype
    #sys.exit(0)
    
    model_comp_spec = np.zeros((total_models, len(model_lam_grid)), dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data
    
    print "All models read."
    
    # Run profiling
    cProfile.runctx("model_mods_cython_copytoedit.do_model_modifications(model_lam_grid, model_comp_spec, \
        resampling_lam_grid, total_models, lsf, obj_photoz)", \
        globals(), locals(), "Profile.prof")
    
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()

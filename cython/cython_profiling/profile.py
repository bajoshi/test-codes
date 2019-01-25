#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.integrate import simps
from scipy.signal import fftconvolve

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
import line_profiler

import matplotlib.pyplot as plt

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
    lsf = np.genfromtxt(lsf_filename)
    
    # ------------------------ Broader LSF --------------------- #
    # using a broader lsf just to see if that can do better
    #interppoints = np.linspace(start=0, stop=lsf_length, num=lsf_length*5, dtype=DTYPE)
    # just making the lsf sampling grid longer # i.e. sampled at more points 
    #broad_lsf = np.interp(interppoints, xp=np.arange(lsf_length), fp=lsf)
    """
    This isn't really making the LSF broader. It is simply adding more points.
    It does give the desired result of smoothing out the model some more but perhaps
    this isn't the right way. You can make the LSF actually broader but
    keep the same number of points.

    What I'm doing below is: 
    1. Fit a gaussian to the LSF and find its std. dev.
    2. Use the fact the (sigma_b)^2 = (sigma_lsf)^2 + (sigma_kernel)^2
    where sigma_b is the std.dev. of the broadened lsf
    sigma_lsf is the std. dev. of the LSF (from guassian fitting above)
    sigma_kernel is the std.dev. of the kernel used to broaden the LSF.
    3. I want the LSF to be broadened by a factor of 1.5x 
    This is because of the mismatch between the pixel scale the LSF
    was measured on vs the pixel scale the grism spectra were drizzled and
    coadded to. It seems liek the LSF pixel scale is 0.033"/pixel and 
    the grism pix scale is 0.05"/pixel from the aXedrizzle conf file.
    So I get 0.05 / 0.033 ~ 1.5x LSF broadening factor.
    """
    # fit
    lsf_length = len(lsf)
    gauss_init = models.Gaussian1D(amplitude=np.max(lsf), mean=lsf_length/2, stddev=lsf_length/4)
    fit_gauss = fitting.LevMarLSQFitter()
    x_arr = np.arange(lsf_length)
    g = fit_gauss(gauss_init, x_arr, lsf)
    # get fit std.dev. and create a gaussian kernel with which to broaden
    kernel_std = 1.118 * g.parameters[2]
    broaden_kernel = Gaussian1DKernel(kernel_std)
    # broaden LSF
    broad_lsf = fftconvolve(lsf, broaden_kernel, mode='same')

    # Code block useful for debugging. Do not remove.
    """
    print g.parameters
    print kernel_std
    print simps(broad_lsf)
    # plot fit and broadened lsf to check
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x_arr, lsf, color='b')
    ax.plot(x_arr, g(x_arr), color='r')
    ax.plot(x_arr, broad_lsf, color='g')

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    sys.exit(0)
    """
    
    # -------------------------------- 
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
    model_lam_grid = model_lam_grid.astype(np.float64)
    
    model_comp_spec = np.zeros((total_models, len(model_lam_grid)), dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data
    
    print "All models read."

    profile = line_profiler.LineProfiler(model_mods_cython_copytoedit.do_model_modifications)
    profile.runcall(model_mods_cython_copytoedit.do_model_modifications, model_lam_grid, model_comp_spec, \
        resampling_lam_grid, total_models, broad_lsf, obj_photoz)
    profile.print_stats()

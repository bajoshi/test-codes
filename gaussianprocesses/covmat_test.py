from __future__ import division

import numpy as np
from astropy.io import fits
#import george
#from george.modeling import Model
#from george import kernels
import emcee
import corner

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')
pears_data = home + '/Documents/PEARS/data_spectra_only/'
massive_galaxies_dir = home + '/Desktop/FIGS/massive-galaxies/'

sys.path.append(massive_galaxies_dir + 'grismz_pipeline/')
import new_refine_grismz_gridsearch_parallel as ngp

def get_covmat(spec_wav, spec_flux, spec_ferr, silent=True):

    # First define the kernel that will be used to model
    # the off-diagonal elements of the covariance matrix.
    galaxy_len_fac = 20
    # galaxy_len_fac includes the effect in correlation due to the 
    # galaxy morphology, i.e., for larger galaxies, flux data points 
    # need to be farther apart to be uncorrelated.
    base_fac = 5
    # base_fac includes the correlation effect due to the overlap 
    # between flux observed at adjacent spectral elements.
    # i.e., this amount of correlation in hte noise will 
    # exist even for a point source
    kern_len_fac = base_fac + galaxy_len_fac
    #kernel = george.GP(kernel=np.var(spec_flux) * kernels.ExpSquaredKernel(kern_len_fac))
    #print "Variance in flux:", np.var(spec_flux)

    """
    kernel.compute(x=spec_wav, yerr=spec_ferr)
    covmat_auto = kernel.get_matrix()
    print kernel.get_parameter_dict()
    print covmat_auto
    print covmat_auto.shape

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(covmat_auto)
    fig.colorbar(cax)
    plt.show()

    sys.exit(0)
    """

    # Get number of spectral elements and define covariance mat
    N = len(spec_wav)
    covmat = np.identity(N)

    # Now populate the elements of the matrix
    len_fac = -1 / (2 * kern_len_fac**2)
    theta_0 = max(spec_ferr)**2
    print "Theta_0 is:", theta_0
    for i in range(N):
        for j in range(N):

            if i == j:
                covmat[i,j] = 1.0/spec_ferr[i]**2
                #print "Exponential factor for element", i, j, "is:", 1.0
            else:
                #print "Exponential factor for element", i, j, "is:", 
                #print np.exp(len_fac * (spec_wav[i] - spec_wav[j])**2)
                covmat[i,j] = (1.0/theta_0) * np.exp(len_fac * (spec_wav[i] - spec_wav[j])**2)

    # PLot 
    if not silent:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel('Spectral element index')
        ax.set_ylabel('Spectral element index')

        #covmat = np.log10(covmat)
        # Set everything below a certain lower limit to exactly zero
        inv_idx = np.where(covmat <= 1e-4 * theta_0)
        covmat[inv_idx] = np.nan
        cax = ax.imshow(covmat)
    
        fig.colorbar(cax)
        plt.show()

    return covmat

def main():

    # Read in a test PEARS grism spectrum
    current_id = 100543
    current_field = 'GOODS-S'
    grism_lam_obs, grism_flam_obs, grism_ferr_obs, pa_chosen, netsig_chosen, return_code = \
    ngp.get_data(current_id, current_field)

    # Tryout a fake spectrum as well
    """
    np.random.seed(12345)
    rng = (-10, 10)
    num = 100
    test_flux = rng[0] + np.diff(rng) * np.sort(np.random.rand(num))
    test_ferr = 0.5 + 0.5 * np.random.rand(num)
    test_wavl = np.arange(num)

    # Plot the spectrum for a quick eyeball check
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.errorbar(test_wavl, test_flux, yerr=test_ferr)
    ax.errorbar(grism_lam_obs, grism_flam_obs, yerr=grism_ferr_obs)
    plt.show()
    sys.exit(0)
    """

    # Get covariance matrix
    get_covmat(grism_lam_obs, grism_flam_obs, grism_ferr_obs)

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)
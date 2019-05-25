from __future__ import division

import numpy as np

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')

sys.path.append(home + '/Desktop/test-codes/gaussianprocesses/')
from gp_george_test import generate_data
from covmat_test import get_covmat

if __name__ == '__main__':
    """
    Will attempt to test cmputation of alpha
    and the subsequent chi2 minimization, while 
    implementing non-zero off-diagonal elements
    in the covariance matrix.
    """

    # Define data and model
    # Using example from george module for gaussian processes
    truth = dict(amp=-5.0, location=0.5, log_sigma2=np.log(1.4))
    N = 100
    x, dat, daterr = generate_data(truth, N)

    # check with a quick plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, dat, yerr=daterr, ecolor='r')
    plt.show()
    """

    # Now compute the covariance matrix for the data
    covmat = get_covmat(x, dat, daterr, silent=True)

    # Now find inverse
    covmat_inv = np.linalg.inv(covmat)

    # again plot and check
    print "Comparison of the product of the covariance matrix and its inverse to the identity matrix",
    print "(should be close to identity):",
    print np.allclose(np.dot(covmat, covmat_inv), np.eye(N))
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    cax1 = ax1.imshow(covmat)
    cax2 = ax2.imshow(covmat_inv)
    fig.colorbar(cax1)
    fig.colorbar(cax2)
    plt.show()
    """

    # Define model
    # For the purposes of comparison I'm simply using 
    # samples drawn from astropy's gaussian + line model.
    # This is simply a numpy array.
    model = np.arange(N)

    # Now compute alpha explicitly
    num = 0
    den = 0  
    for i in range(N):
        for j in range(N):

            num += (dat[i] * model[j] + dat[j] * model[i]) * covmat_inv[i,j]
            if i == j:
                kron_delt = 1
            else:
                kron_delt = 0
            den += model[i] * model[j] * covmat_inv[i,j] * (1 + kron_delt)

    alpha_ = num / den
    print "Alpha computed explicitly:", alpha_

    # Compute alpha using numpy arrays without explicit for loops


    # Compute chi2
    chi2_ = np.matmul((dat - alpha_ * model), np.matmul(covmat_inv, (dat - alpha_ * model)))

    print np.matmul(covmat_inv, (dat - alpha_ * model)).shape
    print chi2_

    # Chi2 and alpha computed without any correlation taken into account
    alpha_nocorr = np.sum(dat * model / (daterr**2)) / np.sum(model**2 / daterr**2)
    chi2_nocorr = np.sum(((dat - alpha_nocorr * model) / daterr)**2)

    print alpha_nocorr
    print chi2_nocorr

    sys.exit(0)




from __future__ import division

import numpy as np
from astropy.modeling import models, fitting

import sys
import os

import matplotlib.pyplot as plt

home = os.getenv('HOME')

sys.path.append(home + '/Desktop/test-codes/gaussianprocesses/')
from gp_george_test import generate_data
from covmat_test import get_covmat

def get_alpha(dat, model, covmat, arr_size):

    num = 0
    den = 0  
    for i in range(arr_size):
        for j in range(arr_size):

            num += (dat[i] * model[j] + dat[j] * model[i]) * covmat[i,j]
            if i == j:
                kron_delt = 1
            else:
                kron_delt = 0
            den += model[i] * model[j] * covmat[i,j] * (1 + kron_delt)

    alpha_ = num / den

    return alpha_

def get_chi2(dat, model, covmat, alpha, arr_size):

    # Compute chi2_ explicitly
    temp_vec = np.zeros(arr_size)
    col_vec = dat - alpha * model
    for u in range(arr_size):
        temp_val = 0.0
        for v in range(arr_size):

            temp_val += covmat[u,v] * col_vec[v]

        temp_vec[u] = temp_val

    chi2_exp = 0.0
    for w in range(arr_size):
        chi2_exp += col_vec[w] * temp_vec[w]

    return chi2_exp

if __name__ == '__main__':
    """
    Will attempt to test cmputation of alpha
    and the subsequent chi2 minimization, while 
    implementing non-zero off-diagonal elements
    in the covariance matrix.
    """

    # Define data and model
    # Using example from george module for gaussian processes
    truth = dict(amp=10.0, location=0.5, log_sigma2=np.log(1.4))
    N = 20
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

    #print "Elements of error array:", daterr
    #print "Elements of 1/err^2 arr:", 1/daterr**2
    #print "Covariance matrix inverse:", "\n", covmat

    """
    # Now find inverse
    covmat_inv = np.linalg.inv(covmat)

    # again plot and check
    print "Comparison of the product of the covariance matrix and its inverse to the identity matrix",
    print "(should be close to identity):",
    print np.allclose(np.dot(covmat, covmat_inv), np.eye(N))
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
    # samples drawn from astropy's gaussian model.
    # This is simply a numpy array.
    gauss_init1 = models.Gaussian1D(amplitude=10.0, mean=0.5, stddev=np.sqrt(1.4))
    model1 = gauss_init1(x)
    gauss_init2 = models.Gaussian1D(amplitude=10.0, mean=0.5, stddev=np.sqrt(5.0))
    model2 = gauss_init2(x)

    fit_gauss = fitting.LevMarLSQFitter()
    g1 = fit_gauss(gauss_init1, x, dat)
    print "\n", "Results from using Astropy fitting:"
    print g1

    # Now compute alpha explicitly
    alpha_1 = get_alpha(dat, model1, covmat, N)
    alpha_2 = get_alpha(dat, model2, covmat, N)
    print "\n", "Alpha 1 computed explicitly:", alpha_1
    print "Alpha 2 computed explicitly:", alpha_2

    # Compute alpha using numpy arrays without explicit nested for loops

    # Get chi2
    #chi2_exp1 = get_chi2(dat, model1, covmat, alpha_1, N)
    #chi2_exp2 = get_chi2(dat, model2, covmat, alpha_2, N)
    #print "Chi2 1 computed explicitly:", chi2_exp1
    #print "Chi2 2 computed explicitly:", chi2_exp2

    # Compute chi2 using matrix multiplication
    chi2_1 = np.matmul((dat - alpha_1 * model1), np.matmul(covmat, (dat - alpha_1 * model1)))
    chi2_2 = np.matmul((dat - alpha_2 * model2), np.matmul(covmat, (dat - alpha_2 * model2)))
    print "Chi2 1 from matrix math:", chi2_1
    print "Chi2 2 from matrix math:", chi2_2

    #### DO complete fitting using chi2 #### 
    # Generate model array
    

    sys.exit(0)

    # Chi2 and alpha computed without any correlation taken into account
    alpha_nocorr = np.sum(dat * model / (daterr**2)) / np.sum(model**2 / daterr**2)
    chi2_nocorr = np.sum(((dat - alpha_nocorr * model) / daterr)**2)

    print alpha_nocorr
    print chi2_nocorr

    sys.exit(0)




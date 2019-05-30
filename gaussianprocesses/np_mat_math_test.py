from __future__ import division

import numpy as np
from astropy.modeling import models, fitting

import sys
import os
import time
import datetime

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
            den += 2 * model[i] * model[j] * covmat[i,j]

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

    # Start time
    start = time.time()
    dt = datetime.datetime
    print "Starting at --", dt.now()

    # Define data and model
    # Using example from george module for gaussian processes
    truth = dict(amp=10.0, location=0.5, log_sigma2=np.log(1.4))
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
    gauss_init1 = models.Gaussian1D(amplitude=2.0, mean=3.45, stddev=0.25)
    model1 = gauss_init1(x)
    gauss_init2 = models.Gaussian1D(amplitude=10.0, mean=0.5, stddev=np.sqrt(5.0))
    model2 = gauss_init2(x)

    fit_gauss = fitting.LevMarLSQFitter()
    g1 = fit_gauss(gauss_init1, x, dat)
    print "\n", "Results from using Astropy fitting:"
    print g1
    
    """
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
    """

    # ------------------------------------------------------ #
    #### DO complete fitting using chi2 #### 
    """
    It should get the mean and stddev correct but it 
    makes sense that it would not get the amplitude 
    right sometimes. This is because the alpha and 
    amplitude are somewhat degenerate -- you can 
    easily shift the gaussian up or down by changing 
    either parameter.

    It should still be okay to use this method for 
    SED fitting though because none of the fitted 
    parameters are so obviously degenerate. There 
    are some minor degeneracies (like age, 
    metallicity, and dust) but we can't do much 
    about that for now.
    """

    # Generate model array
    indiv_param_len = 30
    total_models = indiv_param_len**3
    model_set = np.zeros((total_models, N), dtype=np.float64)

    # define parameter arrya and populate models
    amp_param_arr = np.linspace(1.0, 20.0, indiv_param_len)
    mean_param_arr = np.linspace(0.1, 5.0, indiv_param_len)
    stddev_param_arr = np.linspace(0.5, 2.0, indiv_param_len)

    # Print test params
    if indiv_param_len <= 20:
        print "Amplitude array tested:", amp_param_arr
        print "Mean array tested:", mean_param_arr
        print "Std dev array tested:", stddev_param_arr

    amp_arr = np.zeros(total_models, dtype=np.float64)
    mean_arr = np.zeros(total_models, dtype=np.float64)
    stddev_arr = np.zeros(total_models, dtype=np.float64)

    count = 0
    for i in range(indiv_param_len):
        for j in range(indiv_param_len):
            for k in range(indiv_param_len):

                #print "Initializing model:", count+1
                gauss_init = models.Gaussian1D(amplitude=amp_param_arr[i], mean=mean_param_arr[j], stddev=stddev_param_arr[k])
                model_set[count] = gauss_init(x)
                amp_arr[count] = amp_param_arr[i]
                mean_arr[count] = mean_param_arr[j]
                stddev_arr[count] = stddev_param_arr[k]

                count += 1

    print "\n", "Model set generated. Working on fitting now."
    print "Total time taken up to now--", str("{:.2f}".format(time.time() - start)), "seconds."

    # --------------------------------------------- #
    # Test out vectorized formulae for alpha and chi2
    # This should be a large improvement over explicit for loops.
    out_prod = np.outer(dat, model_set.T.ravel())
    out_prod = out_prod.reshape(N, N, total_models)

    num_vec = np.sum(np.sum(out_prod * covmat[:, :, None], axis=0), axis=0)
    den_vec = np.zeros(total_models)
    alpha_vec = np.zeros(total_models)
    chi2_vec = np.zeros(total_models)
    for i in range(total_models):  # Get rid of this for loop as well, if you can
        den_vec[i] = np.sum(np.outer(model_set[i], model_set[i]) * covmat, axis=None)
        alpha_vec[i] = num_vec[i]/den_vec[i]
        chi2_vec[i] = np.matmul((dat - alpha_vec[i] * model_set[i]), np.matmul(covmat, (dat - alpha_vec[i] * model_set[i])))

    print "Vectorized computations done."
    print "Total time taken up to now--", str("{:.2f}".format(time.time() - start)), "seconds."

    """
    # Using covariance matrix method but with explicit for loops # Takes way too long.
    # Define chi2 and alpha arrays
    alpha_cov = np.zeros(total_models)
    chi2_explicit = np.zeros(total_models)
    chi2_matmul = np.zeros(total_models)

    for i in range(total_models):
        print "Fitting model:", i+1
        alpha_cov[i] = get_alpha(dat, model_set[i], covmat, N)
        #chi2_explicit[i] = get_chi2(dat, model_set[i], covmat, alpha_cov[i], N)
        chi2_matmul[i] = np.matmul((dat - alpha_cov[i] * model_set[i]), np.matmul(covmat, (dat - alpha_cov[i] * model_set[i])))

    # Now check if the alpha and chi2 vectorized computations match the explicit ones
    for i in range(len(alpha_cov)):
        print "Alpha for element", i+1, ":", alpha_cov[i]/alpha_vec[i], "             ",
        print "Chi2  for element", i+1, ":", chi2_matmul[i]/chi2_vec[i]

    #print "Test for closeness of chi2 comptuted explicitly and using matrix math:", np.allclose(chi2_explicit, chi2_matmul)
    print "\n", "Fitting results from min chi2 (with covariance matrix):"
    min_idx = np.argmin(chi2_matmul)
    print "Min index (with covariance matrix):", min_idx
    print "Amplitude (with covariance matrix):", amp_arr[min_idx]
    print "Mean (with covariance matrix):", mean_arr[min_idx]
    print "Std deviation (with covariance matrix):", stddev_arr[min_idx] 
    """

    # --------------------------------------------- # 
    print "\n", "Fitting results from min chi2 (with covariance matrix) using vectorized formulae:"
    min_idx = np.argmin(chi2_vec)
    print "Min index (vectorized with covariance matrix):", min_idx
    print "Amplitude (vectorized with covariance matrix):", amp_arr[min_idx]
    print "Mean (vectorized with covariance matrix):", mean_arr[min_idx]
    print "Std deviation (vectorized with covariance matrix):", stddev_arr[min_idx]

    # --------------------------------------------- # 
    # Compute alpha and chi2 without covariance matrix
    alpha = np.sum(dat * model_set / (daterr**2), axis=1) / np.sum(model_set**2 / daterr**2, axis=1)
    chi2_nocovmat = np.sum(((dat - (alpha * model_set.T).T) / daterr)**2, axis=1)

    min_idx = np.argmin(chi2_nocovmat)
    print "\n", "Fitting results from min chi2 (without covariance matrix):"
    print "Min index (without covariance matrix):", min_idx
    print "Amplitude (without covariance matrix):", amp_arr[min_idx]
    print "Mean (without covariance matrix):", mean_arr[min_idx]
    print "Std deviation (without covariance matrix):", stddev_arr[min_idx]

    print "All done. Total time taken --", str("{:.2f}".format(time.time() - start)), "seconds."
    sys.exit(0)




#cython: profile=True, nonecheck=False
from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now

import numpy as np
cimport numpy as np
from scipy.interpolate import griddata

cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.profile(False)
cdef float simple_mean(np.ndarray[DTYPE_t, ndim=1] a):
    cdef float s = 0 
    cdef int j
    cdef int arr_elem = len(a)
    for j in xrange(0,arr_elem):
        s += a[j]
    return s / arr_elem

@cython.profile(False)
def redshift_and_resample(np.ndarray[DTYPE_t, ndim=2] model_comp_spec_lsfconv, float z, int total_models, \
    np.ndarray[DTYPE_t, ndim=1] model_lam_grid, \
    np.ndarray[DTYPE_t, ndim=1] resampling_lam_grid, int resampling_lam_grid_length):

    cdef float redshift_factor
    cdef int i
    cdef int k
    cdef int q
    cdef list indices
    cdef float lam_step

    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified
    cdef list resampling_lam_grid_weighted

    # --------------- Redshift model --------------- #
    redshift_factor = 1.0 + z
    model_lam_grid *= redshift_factor
    model_comp_spec_lsfconv /= redshift_factor

    # --------------- Do resampling --------------- #
    # Define array to save modified models
    model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    # --------------- Get indices for resampling --------------- #
    # These indices are going to be different each time depending on the redshfit.
    # i.e. Since it uses the redshifted model_lam_grid to get indices.
    indices = []
    ### Zeroth element
    lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
    indices.append(np.where((model_lam_grid >= resampling_lam_grid[0] - lam_step) & (model_lam_grid < resampling_lam_grid[0] + lam_step))[0])

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        indices.append(np.where((model_lam_grid >= resampling_lam_grid[i-1]) & (model_lam_grid < resampling_lam_grid[i+1]))[0])

    ### Last element
    lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
    indices.append(np.where((model_lam_grid >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid < resampling_lam_grid[-1] + lam_step))[0])

    # ---------- Run for loop to resample ---------- #
    for k in range(total_models):

        resampling_lam_grid_weighted = []

        for q in range(resampling_lam_grid_length):

            bin_center = np.sum(model_lam_grid[indices[q]] * model_comp_spec_lsfconv[k][indices[q]]) / np.sum(model_comp_spec_lsfconv[k][indices[q]])
            resampling_lam_grid_weighted.append(bin_center)

            model_comp_spec_modified[k, q] = simple_mean(model_comp_spec_lsfconv[k][indices[q]])
            # Using simple_mean here instead of np.mean gives a BIG improvement (~3x)
            # np.mean probably does all kinds of checks before it computes the actual
            # mean. It also probably works with different datatypes. Since we know that in
            # our case we will always use floats we can easily use this simple_mean function.

        resampling_lam_grid_weighted = np.asarray(resampling_lam_grid_weighted)

    return model_comp_spec_modified
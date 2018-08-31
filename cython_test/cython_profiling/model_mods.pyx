#cython: profile=True, nonecheck=False
from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now

import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float64
DTYPE_c = np.complex128
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t DTYPE_tc

@cython.profile(False)
cdef float simple_mean(np.ndarray[DTYPE_t, ndim=1] a):
    cdef DTYPE_t s = 0 
    cdef int j
    cdef int arr_elem = len(a)
    for j in xrange(0,arr_elem):
        s += a[j]
    return s / arr_elem

def redshift_and_resample(model_comp_spec_lsfconv, float z, int total_models, model_lam_grid, resampling_lam_grid, int resampling_lam_grid_length):

    cdef float redshift_factor
    cdef int i
    cdef int k
    cdef int q
    cdef list indices
    cdef float lam_step

    # --------------- Redshift model --------------- #
    redshift_factor = 1.0 + z
    model_lam_grid_z = model_lam_grid * redshift_factor
    model_comp_spec_redshifted = model_comp_spec_lsfconv / redshift_factor

    # --------------- Do resampling --------------- #
    # Define array to save modified models
    model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    # --------------- Get indices for resampling --------------- #
    # These indices are going to be different each time depending on the redshfit.
    # i.e. Since it uses the redshifted model_lam_grid_z to get indices.
    indices = []
    ### Zeroth element
    lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0])

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0])

    ### Last element
    lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0])

    # ---------- Run for loop to resample ---------- #
    for k in range(total_models):
        for q in range(resampling_lam_grid_length):
            model_comp_spec_modified[k, q] = np.mean(model_comp_spec_redshifted[k][indices[q]])

    return model_comp_spec_modified
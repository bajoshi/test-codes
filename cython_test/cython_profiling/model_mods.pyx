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
    cdef list model_comp_spec_modified_list

    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_redshifted

    # --------------- Redshift model --------------- #
    """
    Seems like I cannot do these operations in place 
    to make them go faster because I've defined the 
    types already above?? Not quite sure... yet.
    """
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
    """
    for k in range(total_models):

        model_comp_spec_modified[k] = [simple_mean(model_comp_spec_redshifted[k][indices[q]]) for q in range(resampling_lam_grid_length)]
        # Using simple_mean here instead of np.mean gives a BIG improvement (~3x)
        # np.mean probably does all kinds of checks before it computes the actual
        # mean. It also probably works with different datatypes. Since we know that in
        # our case we will always use floats we can easily use this simple_mean function.
    """

    model_comp_spec_modified_list = [np.mean(model_comp_spec_redshifted[:, indices[q]], axis=1) for q in range(resampling_lam_grid_length)]
    model_comp_spec_modified = np.asarray(model_comp_spec_modified_list).T

    return model_comp_spec_modified

@cython.profile(False)
def redshift_and_resample_fast(np.ndarray[DTYPE_t, ndim=2] model_comp_spec_lsfconv, float z, int total_models, \
    np.ndarray[DTYPE_t, ndim=1] model_lam_grid, \
    np.ndarray[DTYPE_t, ndim=1] resampling_lam_grid, int resampling_lam_grid_length):

    cdef float redshift_factor
    cdef int i
    cdef int k
    cdef int q
    cdef float lam_step

    """
    Checks to do:
    1.  The definitions here say that these arrays are 2 dimensional,
        however, they say nothing of the shape. Would it go faster if
        I was able to define and shape to force it?
    2.  Can the for loop below that does the resampling be parallelized
        to go faster?
    """

    cdef np.ndarray[long, ndim=1] idx  # Since the indices are all integers
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_redshifted
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified

    # --------------- Redshift model --------------- #
    """
    Seems like I cannot do these operations in place 
    to make them go faster because I've defined the 
    types already above?? Not quite sure... yet.
    """
    redshift_factor = 1.0 + z
    model_lam_grid_z = model_lam_grid * redshift_factor
    model_comp_spec_redshifted = model_comp_spec_lsfconv / redshift_factor

    # --------------- Do resampling --------------- #
    # Define array to save modified models
    model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    ### Zeroth element
    lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
    idx = np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0]
    model_comp_spec_modified[:, 0] = np.mean(model_comp_spec_redshifted[:, idx], axis=1)

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        idx = np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0]
        model_comp_spec_modified[:, i] = np.mean(model_comp_spec_redshifted[:, idx], axis=1)

    ### Last element
    lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
    idx = np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0]
    model_comp_spec_modified[:, -1] = np.mean(model_comp_spec_redshifted[:, idx], axis=1)

    return model_comp_spec_modified
from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now

import numpy as np
cimport numpy as np
from numpy cimport ndarray
from scipy.interpolate import griddata
from joblib import Parallel, delayed

cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef float simple_mean(np.ndarray[DTYPE_t, ndim=1] a):
    cdef float s = 0 
    cdef int j
    cdef int arr_elem = len(a)
    for j in xrange(0,arr_elem):
        s += a[j]
    return s / arr_elem

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

def do_resamp(np.ndarray[DTYPE_t, ndim=2] model_comp_spec_z, np.ndarray[DTYPE_t, ndim=1] model_grid_z, \
    np.ndarray[DTYPE_t, ndim=1] resamp_grid, int p):

    cdef np.ndarray[long, ndim=1] ix

    ix = np.where((model_grid_z >= resamp_grid[p-1]) & (model_grid_z < resamp_grid[p+1]))[0]
    return np.mean(model_comp_spec_z[:, ix], axis=1)

cdef list cy_where_searchrange(np.ndarray[DTYPE_t, ndim=1] a, float low_val, float high_val):
    """
    This simple where function will work only on 1D arrays.
    An analogous function can be constructed for multi-D arrays
    but it is not needed here.

    The structure of this function is optimized to be used
    in this program.
    """

    cdef DTYPE_t [:] a_view = a
    cdef int asize = a_view.shape[0]
    cdef int i
    cdef list where_indices = []

    for i in range(asize):
        if (a_view[i] >= low_val):
            if (a_view[i] < high_val):
                # maybe write this as two if statements rather than one. 
                # Its kinda pythonic at the moment which cython might not like
                where_indices.append(i)

    return where_indices

def redshift_and_resample_fast(np.ndarray[DTYPE_t, ndim=2] model_comp_spec_lsfconv, float z, int total_models, \
    np.ndarray[DTYPE_t, ndim=1] model_lam_grid, \
    np.ndarray[DTYPE_t, ndim=1] resampling_lam_grid, int resampling_lam_grid_length):

    cdef float redshift_factor
    cdef int i
    cdef float lam_step

    """
    Checks to do:
    1.  The definitions here say that these arrays are 2 dimensional,
        however, they say nothing of the shape. Would it go faster if
        I was able to define and shape to force it?
    2.  Can the for loop below that does the resampling be parallelized
        to go faster?
    """

    #cdef np.ndarray[long, ndim=1] idx  # Since the indices are all integers
    cdef list idx
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

    # Seems like you can't pass the above ndarrays directly since they are buffer arrays
    # You have to take memoryviews on them and then pass those.
    cdef np.float64_t [:, :] model_comp_spec_redshifted_view = model_comp_spec_redshifted
    cdef np.float64_t [:] model_lam_grid_z_view = model_lam_grid_z
    cdef np.float64_t [:] resampling_lam_grid_view = resampling_lam_grid

    # --------------- Do resampling --------------- #
    # Define array to save modified models
    model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)
    cdef np.float64_t [:, :] model_comp_spec_modified_view = model_comp_spec_modified

    ### Zeroth element
    lam_step = resampling_lam_grid_view[1] - resampling_lam_grid_view[0]
    #idx = np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0]
    idx = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid[0] - lam_step, resampling_lam_grid[0] + lam_step)
    model_comp_spec_modified_view[:, 0] = np.mean(model_comp_spec_redshifted_view[:, idx], axis=1)

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        #idx = np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0]
        idx = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid[i-1], resampling_lam_grid[i+1])
        model_comp_spec_modified_view[:, i] = np.mean(model_comp_spec_redshifted_view[:, idx], axis=1)

    #model_comp_spec_mod_list = Parallel(n_jobs=3)(delayed(do_resamp)(model_comp_spec_redshifted_view, model_lam_grid_z_view, resampling_lam_grid_view, i) for i in range(1, resampling_lam_grid_length - 1))
    #model_comp_spec_modified[:, 1:-1] = np.asarray(model_comp_spec_mod_list)

    ### Last element
    lam_step = resampling_lam_grid_view[-1] - resampling_lam_grid_view[-2]
    #idx = np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0]
    idx = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid[-1] - lam_step, resampling_lam_grid[-1] + lam_step)
    model_comp_spec_modified_view[:, -1] = np.mean(model_comp_spec_redshifted_view[:, idx], axis=1)

    return model_comp_spec_modified
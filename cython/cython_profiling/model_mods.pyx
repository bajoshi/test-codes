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
    """
    Checks to do:
    1.  The definitions here say that these arrays are 2 dimensional,
        however, they say nothing of the shape. Would it go faster if
        I was able to define and shape to force it?
    2.  Can the for loop below that does the resampling be parallelized
        to go faster?
    """

    # Type definitions
    cdef float redshift_factor
    cdef int i
    cdef int j
    cdef int k
    cdef float lam_step
    cdef float sum_
    cdef int lam_idx
    cdef int num_idx

    cdef int model_lam_grid_len = len(model_lam_grid)

    #cdef np.ndarray[long, ndim=1] idx  # Since the indices are all integers
    cdef list idx
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z = np.zeros(model_lam_grid_len, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_redshifted = np.zeros((total_models, model_lam_grid_len), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    # Views for faster access
    cdef np.float64_t [:] resampling_lam_grid_view = resampling_lam_grid
    cdef np.float64_t [:] model_lam_grid_view = model_lam_grid
    cdef np.float64_t [:, :] model_comp_spec_lsfconv_view = model_comp_spec_lsfconv

    cdef np.float64_t [:] model_lam_grid_z_view = model_lam_grid_z
    cdef np.float64_t [:, :] model_comp_spec_redshifted_view = model_comp_spec_redshifted
    cdef np.float64_t [:, :] model_comp_spec_modified_view = model_comp_spec_modified

    # --------------- Redshift model --------------- #
    """
    Seems like I cannot do these operations in place 
    to make them go faster because I've defined the 
    types already above?? Not quite sure... yet.
    """
    # Doing it with explicit for loops to make it go faster(?)
    redshift_factor = 1.0 + z

    cdef int lamsize = model_lam_grid_view.shape[0]
    
    cdef int xmax = model_comp_spec_lsfconv_view.shape[0]
    cdef int ymax = model_comp_spec_lsfconv_view.shape[1]

    cdef int w
    cdef int x
    cdef int y
    for w in range(lamsize):
        model_lam_grid_z_view[w] = model_lam_grid_view[w] * redshift_factor

    for x in range(xmax):
        for y in range(ymax):
            model_comp_spec_redshifted_view[x, y] = model_comp_spec_lsfconv_view[x, y] / redshift_factor

    # --------------- Do resampling --------------- #
    ### Zeroth element
    lam_step = resampling_lam_grid_view[1] - resampling_lam_grid_view[0]
    #idx = np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0]
    idx = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid_view[0] - lam_step, resampling_lam_grid_view[0] + lam_step)
    num_idx = len(idx)
    for j in range(total_models):

        sum_ = 0
        for k in range(num_idx):
            lam_idx = idx[k]
            sum_ = sum_ + model_comp_spec_redshifted_view[j, lam_idx]

        model_comp_spec_modified_view[j, 0] = sum_ / num_idx

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        #idx = np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0]
        idx = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid_view[i-1], resampling_lam_grid_view[i+1])
        #model_comp_spec_modified_view[:, i] = np.mean(model_comp_spec_redshifted_view[:, idx], axis=1)
        num_idx = len(idx)
        for p in range(total_models):

            sum_ = 0
            for q in range(num_idx):
                lam_idx = idx[q]
                sum_ = sum_ + model_comp_spec_redshifted_view[p, lam_idx]

            model_comp_spec_modified_view[p, i] = sum_ / num_idx

    #model_comp_spec_mod_list = Parallel(n_jobs=3)(delayed(do_resamp)(model_comp_spec_redshifted_view, model_lam_grid_z_view, resampling_lam_grid_view, i) for i in range(1, resampling_lam_grid_length - 1))
    #model_comp_spec_modified[:, 1:-1] = np.asarray(model_comp_spec_mod_list)

    ### Last element
    lam_step = resampling_lam_grid_view[-1] - resampling_lam_grid_view[-2]
    #idx = np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0]
    idx = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid_view[-1] - lam_step, resampling_lam_grid_view[-1] + lam_step)
    num_idx = len(idx)
    for u in range(total_models):

        sum_ = 0
        for v in range(num_idx):
            lam_idx = idx[v]
            sum_ = sum_ + model_comp_spec_redshifted_view[u, lam_idx]

        model_comp_spec_modified_view[u, resampling_lam_grid_length-1] = sum_ / num_idx

    return model_comp_spec_modified
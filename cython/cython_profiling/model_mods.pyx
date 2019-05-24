from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now

import numpy as np
cimport numpy as np
from numpy cimport ndarray
#from scipy.interpolate import griddata
#from joblib import Parallel, delayed

import sys
import os

home = os.getenv('HOME')
sys.path.append(home + '/Desktop/test-codes/gaussianprocesses/')
from covmat_test import get_covmat

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
                where_indices.append(i)

    # Since I had to use a numpy array and initialize it with one zero in it
    # I'm skipping the first zero and returning the rest.
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
    cdef int model_lam_grid_len = len(model_lam_grid)

    #cdef np.ndarray[long, ndim=1] idx  # Since the indices are all integers
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z = np.zeros(model_lam_grid_len, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_redshifted = np.zeros((total_models, model_lam_grid_len), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=np.float64)

    # Views for faster access
    cdef np.float64_t [:] resampling_lam_grid_view = resampling_lam_grid
    #cdef np.float64_t [:] model_lam_grid_view = model_lam_grid
    #cdef np.float64_t [:, :] model_comp_spec_lsfconv_view = model_comp_spec_lsfconv

    cdef np.float64_t [:] model_lam_grid_z_view = model_lam_grid_z
    cdef np.float64_t [:, :] model_comp_spec_redshifted_view = model_comp_spec_redshifted
    cdef np.float64_t [:, :] model_comp_spec_modified_view = model_comp_spec_modified

    # --------------- Redshift model --------------- #
    # Doing it with explicit for loops to make it go faster
    cdef float redshift_factor
    redshift_factor = 1.0 + z

    cdef int lamsize = model_lam_grid.shape[0]
    cdef int xmax = model_comp_spec_lsfconv.shape[0]
    cdef int ymax = model_comp_spec_lsfconv.shape[1]

    cdef int w
    cdef int x
    cdef int y

    # Redshift wavelength
    for w in range(lamsize):
        model_lam_grid_z[w] = model_lam_grid[w] * redshift_factor

    # Redshift fluxes
    for x in range(xmax):
        for y in range(ymax):
            model_comp_spec_redshifted[x, y] = model_comp_spec_lsfconv[x, y] / redshift_factor

    # --------------- Do resampling --------------- #
    # type memory view for indices 
    # Does not work for Python lists. You'll get:
    # TypeError: 'list' does not have the buffer interface
    # So commenting this out until I can figure out a way to use either C or a numpy array.
    #cdef int [:] idx_view
    cdef list idx_view

    # type for loop variables
    cdef unsigned int i, j, k, p, q, u, v

    # Type other variables in the resamp loops below
    cdef float lam_step
    cdef float sum_
    cdef int lam_idx
    cdef int num_idx

    ### Zeroth element
    lam_step = resampling_lam_grid_view[1] - resampling_lam_grid_view[0]
    #idx_view = np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0]
    idx_view = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid_view[0] - lam_step, resampling_lam_grid_view[0] + lam_step)
    num_idx = len(idx_view)
    for j in range(total_models):

        sum_ = 0
        for k in range(num_idx):
            lam_idx = idx_view[k]
            sum_ = sum_ + model_comp_spec_redshifted_view[j, lam_idx]

        model_comp_spec_modified_view[j, 0] = sum_ / num_idx

    ### all elements in between
    for i in range(1, resampling_lam_grid_length - 1):
        #idx_view = np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0]
        idx_view = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid_view[i-1], resampling_lam_grid_view[i+1])
        #model_comp_spec_modified_view[:, i] = np.mean(model_comp_spec_redshifted_view[:, idx_view], axis=1)
        num_idx = len(idx_view)
        for p in range(total_models):

            sum_ = 0
            for q in range(num_idx):
                lam_idx = idx_view[q]
                sum_ = sum_ + model_comp_spec_redshifted_view[p, lam_idx]

            model_comp_spec_modified_view[p, i] = sum_ / num_idx

    # model_comp_spec_mod_list = \
    # Parallel(n_jobs=3)(delayed(do_resamp)(model_comp_spec_redshifted_view, \
    #     model_lam_grid_z_view, resampling_lam_grid_view, i) for i in range(1, resampling_lam_grid_length - 1))
    # model_comp_spec_modified[:, 1:-1] = np.asarray(model_comp_spec_mod_list)

    ### Last element
    lam_step = resampling_lam_grid_view[-1] - resampling_lam_grid_view[-2]
    #idx_view = np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0]
    idx_view = cy_where_searchrange(model_lam_grid_z, resampling_lam_grid_view[-1] - lam_step, resampling_lam_grid_view[-1] + lam_step)
    num_idx = len(idx_view)
    for u in range(total_models):

        sum_ = 0
        for v in range(num_idx):
            lam_idx = idx_view[v]
            sum_ = sum_ + model_comp_spec_redshifted_view[u, lam_idx]

        model_comp_spec_modified_view[u, resampling_lam_grid_length-1] = sum_ / num_idx

    return model_comp_spec_modified

def cy_get_chi2(np.ndarray[DTYPE_t, ndim=1] grism_flam_obs, np.ndarray[DTYPE_t, ndim=1] grism_ferr_obs, np.ndarray[DTYPE_t, ndim=1] grism_lam_obs, \
    np.ndarray[DTYPE_t, ndim=1] phot_flam_obs, np.ndarray[DTYPE_t, ndim=1] phot_ferr_obs, np.ndarray[DTYPE_t, ndim=1] phot_lam_obs, \
    np.ndarray[DTYPE_t, ndim=2] covmat, np.ndarray[DTYPE_t, ndim=2] all_filt_flam_model, np.ndarray[DTYPE_t, ndim=2] model_comp_spec_mod, \
    np.ndarray[DTYPE_t, ndim=1] model_resampling_lam_grid, int total_models):

    # chop the model to be consistent with the objects lam grid
    model_lam_grid_indx_low = np.argmin(abs(model_resampling_lam_grid - grism_lam_obs[0]))
    model_lam_grid_indx_high = np.argmin(abs(model_resampling_lam_grid - grism_lam_obs[-1]))
    model_spec_in_objlamgrid = model_comp_spec_mod[:, model_lam_grid_indx_low:model_lam_grid_indx_high+1]

    # For both data and model, combine grism+photometry into one spectrum.
    # The chopping above has to be done before combining the grism+photometry
    # because todo the insertion correctly the model and grism wavelength
    # grids have to match.

    # Convert the model array to a python list of lists
    # This has to be done because np.insert() returns a new changed array
    # with the new value inserted but I cannot assign it back to the old
    # array because that changes the shape. This works for the grism arrays
    # since I'm simply using variable names to point to them but since the
    # model array is 2D I'm using indexing and that causes the np.insert()
    # statement to throw an error.
    model_spec_in_objlamgrid_list = []
    for j in range(total_models):
        model_spec_in_objlamgrid_list.append(model_spec_in_objlamgrid[j].tolist())

    count = 0
    combined_lam_obs = grism_lam_obs
    combined_flam_obs = grism_flam_obs
    combined_ferr_obs = grism_ferr_obs
    for phot_wav in phot_lam_obs:

        if phot_wav < combined_lam_obs[0]:
            lam_obs_idx_to_insert = 0

        elif phot_wav > combined_lam_obs[-1]:
            lam_obs_idx_to_insert = len(combined_lam_obs)

        else:
            lam_obs_idx_to_insert = np.where(combined_lam_obs > phot_wav)[0][0]

        # For grism
        combined_lam_obs = np.insert(combined_lam_obs, lam_obs_idx_to_insert, phot_wav)
        combined_flam_obs = np.insert(combined_flam_obs, lam_obs_idx_to_insert, phot_flam_obs[count])
        combined_ferr_obs = np.insert(combined_ferr_obs, lam_obs_idx_to_insert, phot_ferr_obs[count])

        # For model
        for i in range(total_models):
            model_spec_in_objlamgrid_list[i] = np.insert(model_spec_in_objlamgrid_list[i], lam_obs_idx_to_insert, all_filt_flam_model[i, count])

        count += 1

    # Convert back to numpy array
    del model_spec_in_objlamgrid  # Trying to free up the memory allocated to the object pointed by the older model_spec_in_objlamgrid
    # Not sure if the del works because I'm using the same name again. Also just not sure of how del exactly works.
    model_spec_in_objlamgrid = np.asarray(model_spec_in_objlamgrid_list)

    # compute alpha and chi2
    alpha_ = np.sum(combined_flam_obs * model_spec_in_objlamgrid / (combined_ferr_obs**2), axis=1) / np.sum(model_spec_in_objlamgrid**2 / combined_ferr_obs**2, axis=1)
    chi2_ = np.sum(((combined_flam_obs - (alpha_ * model_spec_in_objlamgrid.T).T) / combined_ferr_obs)**2, axis=1)

    #print "Min chi2 for redshift:", min(chi2_)

    return chi2_, alpha_
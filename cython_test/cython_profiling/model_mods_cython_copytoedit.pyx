#cython: profile=True, nonecheck=False
from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now

from scipy.signal import fftconvolve
import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.profile(False)
cdef DTYPE_t simple_mean(np.ndarray[DTYPE_t, ndim=1] a):
    cdef DTYPE_t s = 0 
    cdef int j
    cdef int arr_elem = len(a)
    for j in xrange(0,arr_elem):
        s += a[j]
    return s / arr_elem

@cython.profile(False)
cdef np.ndarray[DTYPE_t, ndim=1] simple_fftconvolve(np.ndarray[DTYPE_t, ndim=1] arr, np.ndarray[DTYPE_t, ndim=1] kernel):

    cdef long arr_length = len(arr)
    cdef long kernel_length = len(kernel)
    cdef np.ndarray[DTYPE_t, ndim=1] conv_arr = np.zeros((arr_length + kernel_length - 1), dtype=DTYPE)

    return conv_arr

def do_model_modifications(np.ndarray[DTYPE_t, ndim=1] model_lam_grid, \
    np.ndarray[DTYPE_t, ndim=2] model_comp_spec, np.ndarray[DTYPE_t, ndim=1] resampling_lam_grid, \
    int total_models, np.ndarray[DTYPE_t, ndim=1] lsf, float z):

    # Before fitting
    # 0. get lsf and models (supplied as arguments to this function)
    # 1. redshift the models
    # 2. convolve the models with the lsf
    # 3. resample the models

    # assert types
    assert model_lam_grid.dtype == DTYPE and resampling_lam_grid.dtype == DTYPE
    assert model_comp_spec.dtype == DTYPE and lsf.dtype == DTYPE
    assert type(total_models) is int
    assert type(z) is float

    # Cython type declarations for the variables
    # hardcoded lengths
    # Can len() be redefined as a C function to be faster?
    cdef int resampling_lam_grid_length = len(resampling_lam_grid)
    cdef int lsf_length = len(lsf)
    cdef int model_lam_grid_length = len(model_lam_grid)

    # create empty array in which final modified models will be stored
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified = np.zeros((total_models, resampling_lam_grid_length), dtype=DTYPE)

    # redshift lambda grid for model
    # this is the lambda grid at the model's native resolution
    cdef float redshift_factor = 1.0 + z
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z = model_lam_grid * redshift_factor
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_redshifted = model_comp_spec / redshift_factor

    # more type definitions
    cdef int k
    cdef int i
    cdef int q
    cdef np.ndarray[DTYPE_t, ndim=1] interppoints
    cdef np.ndarray[DTYPE_t, ndim=1] broad_lsf
    cdef np.ndarray[DTYPE_t, ndim=1] temp_broadlsf_model
    cdef np.ndarray[long, ndim=1] new_ind
    cdef np.ndarray[long, ndim=1] idx
    cdef double lam_step
    cdef list indices = []
    
    # --------------- Get indices for resampling --------------- #
    ### Zeroth element
    lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0])

    ### all elements in between
    for i in xrange(1,resampling_lam_grid_length-1):
        indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[i-1]) & (model_lam_grid_z < resampling_lam_grid[i+1]))[0])

    ### Last element
    lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0])

    # --------------- Now loop over all models --------------- #
    for k in xrange(total_models):

        # using a broader lsf just to see if that can do better
        interppoints = np.linspace(start=0, stop=lsf_length, num=lsf_length*5, dtype=DTYPE)
        # just making the lsf sampling grid longer # i.e. sampled at more points 
        broad_lsf = np.interp(interppoints, xp=np.arange(lsf_length), fp=lsf)

        """
        Perhaps you could also chop the model to a smaller wavelength range (chop NOT resample)
        like 1000A to 10000A (models are in rest frame of course) to make the convolution 
        input array smaller and therefore get a speed up.

        #print lsf_length
        #print len(model_comp_spec[0, :])
        #print len(model_comp_spec[:, 0])
        #print np.argmin(abs(model_lam_grid - 1000))
        #print np.argmin(abs(model_lam_grid - 10000))
        #import sys
        #sys.exit(0)

        This 3D casting is currently not giving me the expected 10x speed up within the 
        fftconvolve. Need to check.
        Also need to type the extra variables introduced here.
        """
        # This idea came from Stack Overflow:
        # https://stackoverflow.com/questions/32028979/speed-up-for-loop-in-convolution-for-numpy-3d-array
        # make the kernel and data 3d that does convolution in z axis only
        """
        kernel_3d = np.zeros(shape=(1,1, lsf_length))
        kernel_3d[0, 0, :] = lsf

        data_3d = np.zeros(shape=(1,1, model_lam_grid_length)) 
        data_3d[0, 0, :] = model_comp_spec_view[k]

        temp = fftconvolve(data_3d, kernel_3d, mode='same')
        temp_broadlsf_model = temp[0, 0, :]
        """

        temp_broadlsf_model = fftconvolve(model_comp_spec_redshifted[k], broad_lsf, mode='same')

        model_comp_spec_modified[k] = [simple_mean(temp_broadlsf_model[indices[q]]) for q in xrange(resampling_lam_grid_length)]

    return model_comp_spec_modified

@cython.profile(False)
cdef list simple_where(np.ndarray[DTYPE_t, ndim=1] a, low_val, high_val):
    """
    This simple where function will work only on 1D arrays.
    An analogous function can be constructed for multi-D arrays
    but it is not needed here.

    The structure of this function is optimized to be used
    in this program.

    For now it seems to not make any difference to the speed 
    of the code.
    """

    cdef int a_length = len(a)
    cdef int i
    #cdef np.ndarray[long, ndim=1] where_indices = np.zeros()
    cdef list where_indices = []
    cdef DTYPE_t [:] a_view = a

    for i in range(a_length):
        if (a_view[i] >= low_val) and (a_view[i] < high_val):
            where_indices.append(i)

    return where_indices

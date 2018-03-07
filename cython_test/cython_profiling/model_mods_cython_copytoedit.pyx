#cython: profile=True, nonecheck=False
from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now

from scipy.signal import fftconvolve
import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float64
DTYPE_c = np.complex128
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t DTYPE_tc

@cython.profile(False)
cdef DTYPE_t simple_mean(np.ndarray[DTYPE_t, ndim=1] a):
    cdef DTYPE_t s = 0 
    cdef int j
    cdef int arr_elem = len(a)
    for j in xrange(0,arr_elem):
        s += a[j]
    return s / arr_elem

@cython.profile(False)
cdef np.ndarray[DTYPE_t, ndim=1] simple_1d_fftconvolve(np.ndarray[DTYPE_t, ndim=1] arr, np.ndarray[DTYPE_t, ndim=1] kernel):

    # Define and type lengths
    cdef int arr_length = 6900 #len(arr)
    cdef int kernel_length = len(kernel)
    cdef int output_len = arr_length + kernel_length - 1

    # Figure out the length of the output segments
    # you know that the model array always has length 6900 # so you can eliminate that length operation above as well
    # you also only need to do this once per galaxy NOT once per model and also NOT once per redshift
    # Make 10 segments each of length 690
    cdef int total_segments = 4
    cdef int segment_length = 1725
    cdef int segment_fft_length = segment_length + kernel_length - 1
    cdef int segment_pad_len = segment_fft_length - segment_length
    cdef int kernel_pad_length = segment_fft_length - kernel_length

    # now pad the kernel with zeros
    cdef np.ndarray[DTYPE_t, ndim=1] kernel_padded = np.append(kernel, np.zeros(kernel_pad_length))

    # Define and initialize output array
    cdef np.ndarray[DTYPE_t, ndim=1] conv_arr = np.zeros(output_len, dtype=DTYPE)

    # First get the discrete fourier transform of the kernel
    # make sure that its padded
    cdef np.ndarray[DTYPE_tc, ndim=1] kernel_fft = np.fft.fft(kernel_padded)
    # Divide into real and imag parts
    cdef np.ndarray[DTYPE_t, ndim=1] kernel_fft_real = kernel_fft.real
    cdef np.ndarray[DTYPE_t, ndim=1] kernel_fft_imag = kernel_fft.imag

    # Loop through segments
    # Type intermediate products generated in the loop
    cdef np.ndarray[DTYPE_t, ndim=1] current_seg
    cdef np.ndarray[DTYPE_tc, ndim=1] current_seg_fft
    cdef np.ndarray[DTYPE_t, ndim=1] current_seg_fft_real
    cdef np.ndarray[DTYPE_t, ndim=1] current_seg_fft_imag
    cdef np.ndarray[DTYPE_t, ndim=1] inv_fft_real
    cdef np.ndarray[DTYPE_t, ndim=1] inv_fft_imag
    cdef np.ndarray[DTYPE_tc, ndim=1] inv_fft_array
    cdef np.ndarray[DTYPE_t, ndim=1] conv_seg
    cdef int overlap_length = kernel_length - 1
    cdef np.ndarray[DTYPE_t, ndim=1] overlap = np.zeros(overlap_length)

    cdef int i
    for i in range(total_segments):

        # Get the segment to be convolved
        current_seg = arr[i*segment_length : (i+1)*segment_length]

        # Now pad the slice with zeros
        current_seg = np.append(current_seg, np.zeros(segment_pad_len))

        # get the fft of the slice
        current_seg_fft = np.fft.fft(current_seg)
        current_seg_fft_real = current_seg_fft.real
        current_seg_fft_imag = current_seg_fft.imag

        # get the real and imag parts of the inverse fft
        inv_fft_real = current_seg_fft_real*kernel_fft_real - current_seg_fft_imag*kernel_fft_imag
        inv_fft_imag = current_seg_fft_real*kernel_fft_imag + current_seg_fft_imag*kernel_fft_real

        inv_fft_array = np.array(inv_fft_real + inv_fft_imag*1j, dtype=DTYPE_c)

        # Now get the inverse fft and take the magnitudes of the complex numbers in the array
        # this is because the input is real so the convolved output must also be real
        conv_seg = np.absolute(np.fft.ifft(inv_fft_array))

        # Save the convolution to the final array and 
        # Save the overlap
        conv_arr[i*segment_length : (i+1)*segment_length] += conv_seg[0:segment_length]
        overlap = conv_seg[segment_length:]

        # Now add the overlap to the previous segment
        conv_arr[(i+1)*segment_length:(i+1)*segment_length+overlap_length] += overlap

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
        """

        temp_broadlsf_model = fftconvolve(model_comp_spec_redshifted[k], lsf, mode = 'same')
        model_comp_spec_modified[k] = [simple_mean(temp_broadlsf_model[indices[q]]) for q in xrange(resampling_lam_grid_length)]
        # Do interpolation instead of resampling??
        # This is to remove the dependence on resampling bin edges

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

from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now
import numpy as np
cimport numpy as np

DTYPE = np.float64
DTYPE_c = np.complex128
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t DTYPE_tc

#cdef fft_prep():
#    return 

#cdef np.ndarray[DTYPE_t, ndim=1] simple_1d_fftconvolve(np.ndarray[DTYPE_t, ndim=1] arr, np.ndarray[DTYPE_t, ndim=1] kernel):
def simple_1d_fftconvolve(np.ndarray[DTYPE_t, ndim=1] arr, np.ndarray[DTYPE_t, ndim=1] kernel):

    # Define and type lengths
    cdef int arr_length = len(arr)
    cdef int kernel_length = len(kernel)
    cdef int output_len = arr_length + kernel_length - 1

    # Figure out the length of the output segments
    # you know that the model array always has length 6900 # so you can eliminate that length operation above as well
    # you also only need to do this once per galaxy NOT once per model and also NOT once per redshift
    # Make 10 segments each of length 690
    cdef int total_segments = 10
    cdef int segment_length = 690
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

if __name__ == '__main__':

    print "Use only as module. Exiting."
    import sys
    sys.exit(0)

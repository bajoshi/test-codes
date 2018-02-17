from __future__ import division

from scipy.signal import fftconvolve
import numpy as np
#import numpy.ma as ma

cimport numpy as np
#cimport cython

#from astropy.convolution import convolve_fft
#import matplotlib.pyplot as plt
# check if there is an existing C function to do--
# convolution
# mean
# interpolation
# seems like len() and abs() are already as optimized as they can be
# check the list at http://docs.cython.org/en/latest/src/userguide/language_basics.html#language-basics

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef double simple_mean(np.ndarray[DTYPE_t, ndim=1] a):
    cdef double s = 0 
    cdef int j
    cdef int arr_elem = len(a)
    for j in xrange(0,arr_elem):
        s += a[j]
    return s / arr_elem

def do_model_modifications(np.ndarray[DTYPE_t, ndim=1] model_lam_grid, \
    np.ndarray[DTYPE_t, ndim=2] model_comp_spec, np.ndarray[DTYPE_t, ndim=1] resampling_lam_grid, \
    int total_models, np.ndarray[DTYPE_t, ndim=1] lsf, float z):

    # Before fitting
    # 0. get lsf and models (supplied as arguments to this function)
    # 1. redshift the models
    # 2. convolve the models with the lsf
    # 3. resample the models

    # Cython type declarations for the variables
    # hardcoded lengths
    # Can len() be redefined as a C function to be faster?
    cdef int resampling_lam_grid_length = len(resampling_lam_grid)
    cdef int lsf_length = len(lsf)

    # assert types
    assert model_lam_grid.dtype == DTYPE and resampling_lam_grid.dtype == DTYPE
    assert model_comp_spec.dtype == DTYPE and lsf.dtype == DTYPE
    assert type(total_models) is int
    assert type(z) is float
    #print type(z)
    #if type(z) is DTYPE:
    #    print "All okay here."
    ##assert type(z) is DTYPE
    #z = np.float64(z)
    #print type(z)

    # create empty array in which final modified models will be stored
    cdef np.ndarray[DTYPE_t, ndim=2] model_comp_spec_modified = \
    np.empty((total_models, resampling_lam_grid_length), dtype=DTYPE)

    # redshift lambda grid for model 
    # this is the lambda grid at the model's native resolution
    cdef DTYPE_t redshift_factor = 1.0 + z
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z = model_lam_grid * redshift_factor

    # redshift flux
    model_comp_spec = model_comp_spec / redshift_factor

    # ---------------- Mask potential emission lines ----------------- #
    """
    # Will mask one point on each side of line center i.e. approx 80 A masked
    # These are all vacuum wavelengths
    # first define all variables
    cdef double oiii_5007
    cdef double oiii_4959
    cdef double hbeta
    cdef double hgamma
    cdef double oii_3727

    oiii_5007 = 5008.24
    oiii_4959 = 4960.30
    hbeta = 4862.69
    hgamma = 4341.69
    oii_3727 = 3728.5
    # these two lines (3727 and 3729) are so close to each other 
    # that the line will always blend in grism spectra. 
    # avg wav of the two written here

    # Set up line mask
    cdef np.ndarray[long, ndim=1] line_mask = np.zeros(resampling_lam_grid_length, dtype=np.int)

    # Get redshifted wavelengths and mask
    cdef int oii_3727_idx
    cdef int oiii_5007_idx
    cdef int oiii_4959_idx

    oii_3727_idx = np.argmin(abs(resampling_lam_grid - oii_3727*(1 + z)))
    oiii_5007_idx = np.argmin(abs(resampling_lam_grid - oiii_5007*(1 + z)))
    oiii_4959_idx = np.argmin(abs(resampling_lam_grid - oiii_4959*(1 + z)))

    line_mask[oii_3727_idx-1 : oii_3727_idx+2] = 1
    line_mask[oiii_5007_idx-1 : oiii_5007_idx+2] = 1
    line_mask[oiii_4959_idx-1 : oiii_4959_idx+2] = 1
    """

    # more type definitions
    cdef int k
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] interppoints
    cdef np.ndarray[DTYPE_t, ndim=1] broad_lsf
    cdef np.ndarray[DTYPE_t, ndim=1] temp_broadlsf_model
    cdef np.ndarray[DTYPE_t, ndim=1] resampled_flam_broadlsf
    cdef np.ndarray[long, ndim=1] new_ind
    cdef np.ndarray[long, ndim=1] idx
    cdef double lam_step_low
    cdef double lam_step_high
    cdef double lam_step
    cdef list indices = []
    #cdef np.ndarray[long, ndim=2] indices
    
    for i in xrange(1,resampling_lam_grid_length-1):

        lam_step_high = resampling_lam_grid[i+1] - resampling_lam_grid[i]
        lam_step_low = resampling_lam_grid[i] - resampling_lam_grid[i-1]

        indices.append(np.where((model_lam_grid_z >= resampling_lam_grid[i] - lam_step_low) & \
            (model_lam_grid_z < resampling_lam_grid[i] + lam_step_high))[0])

    for k in xrange(total_models):

        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #ax1.plot(model_lam_grid_z, model_comp_spec[k])
        #ax1.set_xlim(5000, 10500)

        #model_comp_spec[k] = convolve_fft(model_comp_spec[k], lsf)#, boundary='extend')
        # seems like boundary='extend' is not implemented 
        # currently for convolve_fft(). It works with convolve() though.

        # using a broader lsf just to see if that can do better
        #interppoints = np.linspace(start=0, stop=lsf_length, num=lsf_length*10, dtype=DTYPE)
        # just making the lsf sampling grid longer # i.e. sampled at more points 
        #broad_lsf = np.interp(interppoints, xp=np.arange(lsf_length), fp=lsf)
        temp_broadlsf_model = fftconvolve(model_comp_spec[k], lsf)

        #ax2.plot(model_lam_grid_z, temp_broadlsf_model)
        #ax2.set_xlim(5000, 10500)

        # resample to object resolution
        resampled_flam_broadlsf = np.zeros(resampling_lam_grid_length, dtype=DTYPE)

        ### Zeroth element
        lam_step = resampling_lam_grid[1] - resampling_lam_grid[0]
        idx = np.where((model_lam_grid_z >= resampling_lam_grid[0] - lam_step) & \
            (model_lam_grid_z < resampling_lam_grid[0] + lam_step))[0]
        resampled_flam_broadlsf[0] = simple_mean(temp_broadlsf_model[idx])

        ### Last element
        lam_step = resampling_lam_grid[-1] - resampling_lam_grid[-2]
        idx = np.where((model_lam_grid_z >= resampling_lam_grid[-1] - lam_step) & \
            (model_lam_grid_z < resampling_lam_grid[-1] + lam_step))[0]
        resampled_flam_broadlsf[-1] = simple_mean(temp_broadlsf_model[idx])

        ### all elements in between
        resampled_flam_broadlsf[1:resampling_lam_grid_length-1] = \
        [simple_mean(temp_broadlsf_model[indices[i-1]]) for i in xrange(1,resampling_lam_grid_length-1)]
        # I use i-1 in indices because the xrange starts from 1

        # Now mask the flux at these wavelengths using the mask generated before the for loop
        #model_comp_spec_modified[k] = ma.array(resampled_flam_broadlsf, mask=line_mask)
        model_comp_spec_modified[k] = resampled_flam_broadlsf

        #ax3.plot(resampling_lam_grid, resampled_flam_broadlsf)
        #ax3.set_xlim(5000, 10500)

        #plt.show()
        #plt.cla()
        #plt.clf()
        #plt.close()

    return model_comp_spec_modified

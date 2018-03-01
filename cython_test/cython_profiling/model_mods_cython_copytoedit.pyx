#cython: boundscheck=False, nonecheck=False
#cython: profile=True
from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now

from scipy.signal import fftconvolve
import numpy as np
#import numpy.ma as ma

cimport numpy as np
cimport cython

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

@cython.profile(False)
cdef double simple_mean(np.ndarray[DTYPE_t, ndim=1] a):
    cdef double s = 0 
    cdef int j
    cdef int arr_elem = len(a)
    for j in xrange(0,arr_elem):
        s += a[j]
    return s / arr_elem

@cython.profile(False)
cdef list simple_where(np.ndarray[DTYPE_t, ndim=1] a, low_val, high_val):
    """
    This simple where function will work only on 1D arrays.
    An analogous function can be constructed for multi-D arrays
    but its not needed here.

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
    cdef int model_lam_grid_length = len(model_lam_grid)

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
    np.zeros((total_models, resampling_lam_grid_length), dtype=DTYPE)

    # Views 
    cdef DTYPE_t [:, :] model_comp_spec_view = model_comp_spec
    cdef DTYPE_t [:] resampling_lam_grid_view = resampling_lam_grid
    #cdef DTYPE_t [:, :] model_comp_spec_modified_view = model_comp_spec_modified

    # redshift lambda grid for model
    # this is the lambda grid at the model's native resolution
    cdef float redshift_factor = 1.0 + z
    cdef np.ndarray[DTYPE_t, ndim=1] model_lam_grid_z = np.zeros(model_lam_grid_length)

    #cdef int w
    #for w in range(model_lam_grid_length):
    #    model_lam_grid_z_view[w] = model_lam_grid_view[w] * redshift_factor
    model_lam_grid_z = model_lam_grid * redshift_factor
    #cdef DTYPE_t [:] model_lam_grid_z_view = model_lam_grid_z

    # redshift flux
    cdef int u
    cdef int v
    for u in range(model_comp_spec_view.shape[0]):
        for v in range(model_comp_spec_view.shape[1]):
            model_comp_spec_view[u,v] /= redshift_factor
    # using /= instead of using the longer form is faster
    # i.e. the longer form --> var = var / denominator copies the 'var' I think
    # the short hand modifies it in place which is why it is faster

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
    #cdef np.ndarray[DTYPE_t, ndim=1] resampled_flam_broadlsf
    cdef np.ndarray[long, ndim=1] new_ind
    cdef np.ndarray[long, ndim=1] idx
    #cdef double lam_step_low
    #cdef double lam_step_high
    cdef double lam_step
    cdef list indices = []
    
    # --------------- Get indices for resampling --------------- #
    ### Zeroth element
    lam_step = resampling_lam_grid_view[1] - resampling_lam_grid_view[0]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid_view[0] - lam_step) & \
        (model_lam_grid_z < resampling_lam_grid_view[0] + lam_step))[0])
    #indices.append(simple_where(model_lam_grid_z, resampling_lam_grid_view[0] - lam_step, \
    #    resampling_lam_grid_view[0] + lam_step))

    ### all elements in between
    for i in xrange(1,resampling_lam_grid_length-1):

        #lam_step_high = resampling_lam_grid_view[i+1] - resampling_lam_grid_view[i]
        #lam_step_low = resampling_lam_grid_view[i] - resampling_lam_grid_view[i-1]
        indices.append(np.where((model_lam_grid_z >= resampling_lam_grid_view[i-1]) & \
            (model_lam_grid_z < resampling_lam_grid_view[i+1]))[0])
        #indices.append(simple_where(model_lam_grid_z, resampling_lam_grid_view[i-1], resampling_lam_grid_view[i+1]))

    ### Last element
    lam_step = resampling_lam_grid_view[-1] - resampling_lam_grid_view[-2]
    indices.append(np.where((model_lam_grid_z >= resampling_lam_grid_view[-1] - lam_step) & \
        (model_lam_grid_z < resampling_lam_grid_view[-1] + lam_step))[0])
    #indices.append(simple_where(model_lam_grid_z, resampling_lam_grid_view[-1] - lam_step, \
    #    resampling_lam_grid_view[-1] + lam_step))

    # --------------- Now loop over all models --------------- #
    for k in xrange(total_models):

        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #ax1.plot(model_lam_grid_z_view, model_comp_spec[k])
        #ax1.set_xlim(5000, 10500)

        #model_comp_spec[k] = convolve_fft(model_comp_spec[k], lsf)#, boundary='extend')
        # seems like boundary='extend' is not implemented 
        # currently for convolve_fft(). It works with convolve() though.

        # using a broader lsf just to see if that can do better
        #interppoints = np.linspace(start=0, stop=lsf_length, num=lsf_length*10, dtype=DTYPE)
        # just making the lsf sampling grid longer # i.e. sampled at more points 
        #broad_lsf = np.interp(interppoints, xp=np.arange(lsf_length), fp=lsf)

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

        temp_broadlsf_model = fftconvolve(model_comp_spec_view[k], lsf, mode='same')

        #ax2.plot(model_lam_grid_z_view, temp_broadlsf_model)
        #ax2.set_xlim(5000, 10500)

        # resample to object resolution
        #resampled_flam_broadlsf = np.zeros(resampling_lam_grid_length, dtype=DTYPE)

        ### all elements in between
        model_comp_spec_modified[k] = \
        [simple_mean(temp_broadlsf_model[indices[i]]) for i in xrange(resampling_lam_grid_length)]
        # I use i-1 in indices because the xrange starts from 1

        # Now mask the flux at these wavelengths using the mask generated before the for loop
        #model_comp_spec_modified[k] = ma.array(resampled_flam_broadlsf, mask=line_mask)
        #model_comp_spec_modified[k] = resampled_flam_broadlsf

        #ax3.plot(resampling_lam_grid, resampled_flam_broadlsf)
        #ax3.set_xlim(5000, 10500)

        #plt.show()
        #plt.cla()
        #plt.clf()
        #plt.close()

    return model_comp_spec_modified

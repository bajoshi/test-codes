from __future__ import division
# Tunring on cdivision seems to make no difference to the speed as of now
import numpy as np
cimport numpy as np

from scipy.signal import fftconvolve

home = os.getenv('HOME')
pears_datadir = home + '/Documents/PEARS/data_spectra_only/'
stacking_analysis_dir = home + "/Desktop/FIGS/stacking-analysis-pears/"
massive_galaxies_dir = home + "/Desktop/FIGS/massive-galaxies/"
savefits_dir = home + "/Desktop/FIGS/new_codes/bc03_fits_files_for_refining_redshifts/"
lsfdir = home + "/Desktop/FIGS/new_codes/pears_lsfs/"
figs_dir = home + "/Desktop/FIGS/"

cdef np.ndarray[DTYPE_t, ndim=1] simple_1d_fftconvolve(np.ndarray[DTYPE_t, ndim=1] arr, np.ndarray[DTYPE_t, ndim=1] kernel):

    # Define lengths
    cdef long arr_length = len(arr)
    cdef long kernel_length = len(kernel)
    cdef long output_len = arr_length + kernel_length - 1

    # Define and initialize output array
    cdef np.ndarray[DTYPE_t, ndim=1] conv_arr = np.zeros(output_len, dtype=DTYPE)

    # First get the discrete fourier transform of the kernel
    cdef np.ndarray[np.complex128, ndim=1] kernel_fft = np.fft.rfft(kernel)
    


    return conv_arr

if __name__ == '__main__':

    # read in entire model set
    bc03_all_spec_hdulist = fits.open(figs_dir + 'all_comp_spectra_bc03_ssp_and_csp_nolsf_noresample.fits')
    total_models = 34542
    
    # arrange the model spectra to be compared in a properly shaped numpy array for faster computation
    example_filename_lamgrid = 'bc2003_hr_m22_tauV20_csp_tau50000_salp_lamgrid.npy'
    bc03_galaxev_dir = home + '/Documents/GALAXEV_BC03/'
    model_lam_grid = np.load(bc03_galaxev_dir + example_filename_lamgrid)
    model_lam_grid = model_lam_grid.astype(np.float64)
    
    model_comp_spec = np.zeros((total_models, len(model_lam_grid)), dtype=np.float64)
    for j in range(total_models):
        model_comp_spec[j] = bc03_all_spec_hdulist[j+1].data
    
    print "All models read."

    # Read in LSF
    lsf_filename = lsfdir + "south_lsfs/" + "s" + str(current_id) + "_" + pa_chosen.replace('PA', 'pa') + "_lsf.txt"
    lsf = np.genfromtxt(lsf_filename)

    # Now check the FFT convolution of these models your way and the Scipy way
    # They should be the same
    for k in range(total_models):

        conv_model = simple_1d_fftconvolve(model_comp_spec[k], lsf)

        conv_model_scipy = fftconvolve(model_comp_spec[k], lsf, mode='same')

        print len(model_lam_grid)
        print len(conv_model)
        print len(conv_model_scipy)

        # PLot to compare
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(model_lam_grid, conv_model, color='b')
        ax.plot(model_lam_grid, conv_model_scipy, color='r')

        plt.show()

        sys.exit(0)


"""
This code doesn't work yet.
"""

import pstats, cProfile

import pyximport
pyximport.install()

import test_model_mods_cython

cProfile.runctx("test_model_mods_cython.do_model_modifications(lam_obs, model_lam_grid, model_comp_spec, resampling_lam_grid, total_models, lsf, z)", \
	globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
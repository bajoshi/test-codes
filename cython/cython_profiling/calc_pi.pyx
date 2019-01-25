# encoding: utf-8
# cython: profile=True
# filename: calc_pi.pyx

cimport cython

# check the annotated version of the code to see what happens if the cdivision line is commented out
# If you turn on cdivision then it will stop Python from checking for ZeroDivisionError which 
# could speed things up. The cdivision verison of this inline function does not have any yellow lines.
# If the cdivision is commented out then you'll see yellow lines.
@cython.cdivision(True)
@cython.profile(False)
cdef inline double recip_square(int i):
    return 1./i*i

def approx_pi(int n=10000000):
    cdef double val = 0.
    cdef int k
    for k in xrange(1,n+1):
        val += recip_square(k)
    return (6 * val)**.5
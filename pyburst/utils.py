## Copyright (C) 2019 Eric Chassande-Mottin, CNRS (France)
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see .

import math
import numpy
from scipy.signal import upfirdn

# Default parameters for the interpolator filter
LOG10_REJECTION = -3.0
REJECTION_DB = -20 * LOG10_REJECTION

STOPBAND_CUTOFF_F = 0.5
ROLL_OFF_WIDTH = STOPBAND_CUTOFF_F / 10  
## determine filter length
## use empirical formula from [1] Chap 7, Eq. (7.63) p 476
HALF_INTERP_FILTER_LENGTH = math.ceil((REJECTION_DB-8.0) / (28.714 * ROLL_OFF_WIDTH))
INTERP_FILTER_LENGTH = 2 * HALF_INTERP_FILTER_LENGTH + 1

def next_odd(n):
    """ Compute the next odd number
    """
    return math.ceil(n) // 2 * 2 + 1

def pseudoinverse(a, rcond=1e-15):
    """
    Compute pseudoinverse of 2D array corresponding to the last two dimensions
    of the input array using the SVD
    """
    swap = numpy.arange(a.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]

    u, s, v = numpy.linalg.svd(a)
    cutoff = numpy.maximum.reduce(s, axis=-1, keepdims=True) * rcond

    mask = s > cutoff
    s[mask] = 1. / s[mask]
    s[~mask] = 0

    return numpy.einsum('...uv,...vw->...uw',
                     numpy.transpose(v, swap) * s[..., None, :],
                     numpy.transpose(u, swap))

def frac_time_shift(x, shift, interp_filt=None):
        """ 
        Shift the input time series by a (possibly fractional) number of samples. 
        The interpolator filter is either specified or either designed with a Kaiser-windowed 
        sinecard.

        x: input timeseries (Numpy array)
        shift: shift in (possibly non-integer) number of samples (scalar)
        interp_filt: interpolator filter (Numpy array -- default = None)

        Ref [1] A. V. Oppenheim, R. W. Schafer and J. R. Buck, Discrete-time signal 
        processing, Signal processing series, Prentice-Hall, 1999

        Ref [2] T.I. Laakso, V. Valimaki, M. Karjalainen and U.K. Laine Splitting the 
        unit delay, IEEE Signal Processing Magazine, vol. 13, no. 1, pp 30--59 Jan 1996
        """

        if float(shift).is_integer():
            return numpy.roll(x, int(shift))

        int_shift = int(numpy.fix(shift))
        frac_shift = shift - int_shift

        if interp_filt is None:

            # Compute the ideal (sinecard) interpolation filter
            time = numpy.arange(-HALF_INTERP_FILTER_LENGTH, HALF_INTERP_FILTER_LENGTH+1)
            sinc_filter = 2 * STOPBAND_CUTOFF_F * \
                              numpy.sinc(2 * STOPBAND_CUTOFF_F * (time-frac_shift))

            # Window ideal (sincard) filter
            interp_filt = sinc_filter * \
                            numpy.kaiser(INTERP_FILTER_LENGTH, _kaiser_beta(REJECTION_DB))

        offset = int(len(interp_filt)/2)
                              
        # Pre and postpad filter response
        interp_filt = numpy.pad(interp_filt, \
                                (len(interp_filt), len(x)  + offset))

        # Filtering
        xfilt = upfirdn(x, interp_filt , 1, 1)

        return numpy.roll(xfilt[offset+1:offset+1+len(x)], int_shift)
                      
def _kaiser_beta(rejection_db):
    """
    Determine the beta parameter for the Kaiser taper window
    using the empirical formula from Ref [1] Chap 7, Eq. (7.62) p 474
    """
                              
    if rejection_db >= 21 and rejection_db <= 50:
        return 0.5842 * (rejection_db - 21)**0.4 + 0.07886 * (rejection_db - 21)
    elif rejection_db > 50:
        return 0.1102 * (rejection_db - 8.7)
    else:
        return 0.0


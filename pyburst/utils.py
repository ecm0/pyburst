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
# numpy.fft import fft, ifftshift, ifft

import scipy.signal, scipy.special, scipy.interpolate

## frac time shift: default parameters for the interpolator filter
LOG10_REJECTION = -3.0
REJECTION_DB = -20 * LOG10_REJECTION

STOPBAND_CUTOFF_F = 0.5
ROLL_OFF_WIDTH = STOPBAND_CUTOFF_F / 10  

## To determine the filter length, we use
## the empirical formula from [1] Chap 7, Eq. (7.63) p 476
HALF_INTERP_FILTER_LENGTH = math.ceil((REJECTION_DB-8.0) / (28.714 * ROLL_OFF_WIDTH))
INTERP_FILTER_LENGTH = 2 * HALF_INTERP_FILTER_LENGTH + 1

## delayseq: empirical roll-off and rejection
## for the FFT-based interpolation filter
DELAYSEQ_ROLL_OFF = 0.1553
DELAYSEQ_LOG10_REJECTION = -3.0

def next_odd(n):
    """ Compute the next odd number
    """
    return math.ceil(n) // 2 * 2 + 1

def next_pow2(n):
    """ Compute the next power of two
    """
    return 2 ** (int(n) - 1).bit_length()

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

        if interp_filt is None:
            interp_filt = _design_default_filter(shift - int_shift)

        # Using numpy.convolve as this is the fastest convolution method
        # for small number of taps according to
        # https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
        return numpy.roll(numpy.convolve(x, interp_filt, mode='same'), int_shift)

        # Earlier implementation using polyphase resampling
        #
        # Pre and postpad filter response
        # offset = int(len(interp_filt)/2)
        # interp_filt = numpy.pad(interp_filt, (0, len(x) - len(interp_filt) + offset))
        #
        # Filtering
        # xfilt = scipy.signal.upfirdn(x, interp_filt, 1, 1)
        #
        # Select useful part of the data and apply integer shift
        #return numpy.roll(xfilt[(offset):(len(x)+offset)], int_shift)

    
def _design_default_filter(shift, length=INTERP_FILTER_LENGTH, stopband_cutoff_f=STOPBAND_CUTOFF_F, rejection_db=REJECTION_DB):
    """
    Design the interpolation filter
    """

    if numpy.abs(shift) > 1:
        shift = shift - numpy.fix(shift)

    # Change length to next odd number
    if length % 2 == 0:
        length = length + 1

    half_length = (length-1)/2
        
    # Compute the ideal (sinecard) interpolation filter
    time = numpy.arange(-half_length, half_length+1)
    sinc_filter = 2 * stopband_cutoff_f * \
                  numpy.sinc(2 * stopband_cutoff_f * (time - shift))

    # Compute taper window
    window = _kaiser(length, _kaiser_beta(rejection_db), shift)
    # window = scipy.interpolate.interp1d(time, kaiser,kind='cubic',fill_value='extrapolate')(time + shift)

    return sinc_filter * window
    
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

def _kaiser(length, beta, shift=0):
    """
    Compute a Kaiser window with parameter beta and possibly 
    non-integer shift
    """

    # Generate shifted x-axis to compute the Kaiser window
    x = numpy.arange(length, dtype=numpy.complex64) - shift
    # Set the type to complex valued as the sqrt in the 
    # following computation can lead to imaginary numbers
    x = 2 * beta / (length-1) * numpy.sqrt(x * (length-1 - x))
    # Use scipy.special.iv instead of numpy.i0 because this function
    # is able to handle complex numbers
    return numpy.real(scipy.special.iv(0,x)/scipy.special.iv(0,beta))

def delayseq(x, shift):
    """
    Shift the input time series by a (possible fractional) number of samples
    using an frequency-domain algorithm that applies frequency-dependent phase shift
    after transforming the input signal using FFT

    x: input timeseries (Numpy array)
    shift: shift in (possibly non-integer) number of samples (scalar)
    """
    
    if float(shift).is_integer():
        return numpy.roll(x, int(shift))

    int_shift = int(numpy.fix(shift))
    frac_shift = shift - int_shift
    
    nfft = next_pow2(x.size + int_shift)
    freqs = 2 * math.pi * numpy.fft.ifftshift((numpy.arange(nfft) - nfft // 2)) / nfft
    tmp = numpy.fft.ifft(numpy.fft.fft(x, nfft)  \
                                     * numpy.exp(-1j * frac_shift * freqs)).real

    return numpy.roll(tmp[:x.size], int_shift)

def angle_between(v1, v2):
    "Compute the angle between two vectors in radians"

    if numpy.linalg.norm(v1) == 0 or \
       numpy.linalg.norm(v2) == 0:
        return None
    
    # normalize to unit norm
    v1 /= numpy.linalg.norm(v1)
    v2 /= numpy.linalg.norm(v2)

    # clip is required to handle exactly coaligned vectors
    # return numpy.arccos(numpy.clip(numpy.dot(v1, v2), -1.0, 1.0))
    return numpy.arctan2(numpy.linalg.norm(numpy.cross(v1, v2)), numpy.dot(v1, v2))

def orthonormalize(v1, v2, dominant_polar_frame=False):
    """
    Orthonormalize the input vectors
    
    v1, v2: input vectors (Numpy array)
    dominant_polar_frame: when True, the returned basis is the
    major and minor axes of the ellipse generated by 
    v1 and v2 (default: False)
    """
    
    assert v1.ndim == 1 and v2.ndim == 1, "Input vars should be one-dim vectors"
    assert v1.shape == v2.shape, "Input vectors should have same shape"

    v = numpy.column_stack((v1, v2))
    
    if dominant_polar_frame is True:

        # This is the standard calculation in the two-dim case :
        # psi = .5 * numpy.arctan2(2* (v1 @ v2), (v1 @ v1) - (v2 @ v2))
        # Compute the rotation matrix that aligns with the major and minor
        # axes of the ellipse generated by v1 and v2
        # c, s = math.cos(psi), math.sin(psi)
        # R = numpy.array(((c, -s), (s, c)))
        # res = v @ R

        # Instead we use the SVD:
        u, s, vh = numpy.linalg.svd(v)

        orthobasis = (u[:, 0], u[:, 1])
        scalings = (s[0], s[1])
        
    else: 
        ## Apply Gram-Schmidt orthonormalization by using QR factorization
        res, _ = numpy.linalg.qr(v)

        orthobasis = (res[:, 0]/numpy.linalg.norm(res[:,0]), \
                      res[:, 1]/numpy.linalg.norm(res[:,1]))
        scalings = (numpy.linalg.norm(res[:,0]), \
                    numpy.linalg.norm(res[:,1]))
         
    return orthobasis, scalings

from unittest import TestCase

import math
import numpy
from scipy.signal import resample_poly

import pyburst.utils as pb

def test_next_odd():
    """ Test next_odd func """
    
    TestCase.assertEqual(next_odd(2), 3)
    TestCase.assertEqual(next_odd(3), 3)
    TestCase.assertEqual(next_odd(4), 5)

def sinegauss(time, sigma, f, phi):
    return numpy.exp(-time**2/(2*sigma)) * \
        numpy.sin(2 * math.pi * f * time + phi)
    
def test_frac_time_shift_with_sinus():
    """ Test frac_time_shift with sinusoidal signal"""

    N = 1024
    delays = (1.5, 7/6, math.rand) 
    time = numpy.arange(N)
    time_shifted = time + delay
    sigma = N/4

    for delay in delays:
        for f in numpy.arange(N/2)/N:
            phi =  2 * math.pi * math.rand
            signal = sinegauss(time, sigma, f, phi)
            estimate_shifted = frac_time_shift(signal, delay)
            exact_shifted = sinegauss(time_shifted, sigma, f, phi)
            error = numpy.max(numpy.abs(estimate_shifted-exact_shifted))
        
            TestCase.assertLess(error, 10**pb.LOG10_REJECTION)

def test_frac_time_shift_with_random():
    """ Test frac_time_shift with a random signal"""

    N = 1024
    p = 6
    q = 7
    b, a = scipy.signal.butter(10,.25)
    delays = (1.5, 7/6, math.rand) 

    for delay in delays:

        noise = numpy.pad(numpy.random.rand(N/2), (N/4, N/4))
        filt_noise = filtfilt(b, a, noise)

        res1 = frac_time_shift(resample_poly(filt_noise, p, q), delay)
        res2 = resample_poly(frac_time_shift(filt_noise, delay) p, q)
        error = numpy.max(numpy.abs(estimate_shifted-exact_shifted))

        TestCase.assertLess(error, 10**pb.LOG10_REJECTION)

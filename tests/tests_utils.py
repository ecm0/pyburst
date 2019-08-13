from unittest import TestCase

import math
import numpy
import scipy.signal

import pyburst.utils as pb

def sinegauss(time, sigma, f, phi):
    return numpy.exp(-time**2/(2*sigma)) * \
        numpy.sin(2 * math.pi * f * time + phi)

class TestUtils(TestCase): 

    def test_next_odd(self):
        """ Test next_odd func """
        
        self.assertEqual(pb.next_odd(2), 3)
        self.assertEqual(pb.next_odd(3), 3)
        self.assertEqual(pb.next_odd(4), 5)
    
    def test_frac_time_shift_with_sinus(self):
        """ Test frac_time_shift with sinusoidal signal"""

        N = 1024
        delays = (1.5, 7/6, numpy.random.rand()) 
        time = numpy.arange(N)
        sigma = N/4

        for delay in delays:
            time_shifted = time + delay
            
            for f in numpy.arange(N/2)/N:
                phi =  2 * math.pi * numpy.random.rand()
                signal = sinegauss(time, sigma, f, phi)
                estimate_shifted = pb.frac_time_shift(signal, delay)
                exact_shifted = sinegauss(time_shifted, sigma, f, phi)
                error = numpy.max(numpy.abs(estimate_shifted-exact_shifted))
                
                self.assertLess(error, 10**pb.LOG10_REJECTION)

    def test_frac_time_shift_with_random(self):
        """ Test frac_time_shift with a random signal"""
        
        N = 1024
        p = 6
        q = 7
        b, a = scipy.signal.butter(10,.25)
        delays = (1.5, 7/6, numpy.random.rand()) 
        
        for delay in delays:
            
            noise = numpy.pad(numpy.random.rand(N//2), (N//4, N//4))
            filt_noise = scipy.signal.filtfilt(b, a, noise)

            res1 = pb.frac_time_shift(scipy.signal.resample_poly(filt_noise, p, q), delay)
            res2 = scipy.signal.resample_poly(pb.frac_time_shift(filt_noise, delay), p, q)
            error = numpy.max(numpy.abs(res1-res2))

            self.assertLess(error, 10**pb.LOG10_REJECTION)

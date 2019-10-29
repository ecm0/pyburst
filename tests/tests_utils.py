from unittest import TestCase

import math
import numpy
import scipy.signal
import pyburst.utils as pb

import matplotlib.pyplot as plt

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
        shifts = (0.5, 1/6, numpy.random.uniform(low=-5, high=+5))
        time = numpy.arange(-N/2,N/2)
        sigma = N/4

        for shift in shifts:

            time_shifted = time - shift
            # error = []

            for f in numpy.arange(math.floor((pb.STOPBAND_CUTOFF_F-pb.ROLL_OFF_WIDTH) * N))/N:
                phi =  2 * math.pi * numpy.random.rand()
                signal = sinegauss(time, sigma, f, phi)
                estimate_shifted = pb.frac_time_shift(signal, shift)
                exact_shifted = sinegauss(time_shifted, sigma, f, phi)
                error = numpy.max(numpy.abs(estimate_shifted-exact_shifted))
                # error.append(numpy.max(numpy.abs(estimate_shifted-exact_shifted)))
                self.assertLess(error, 10**pb.LOG10_REJECTION)
                
            # ax1 = plt.subplot(2, 2, 1)
            # ax1.semilogy(error)
            # ax1.grid()
            #
            # ax2 = plt.subplot(2, 2, 2)
            # plt.plot(signal, label='orig')
            # ax2.plot(estimate_shifted, label='shift')
            # ax2.plot(exact_shifted, '.', label='exact')
            # ax2.grid()
            # plt.plot(numpy.abs(estimate_shifted-exact_shifted), label='error')
            # ax2.legend()
            #
            # ax3 = plt.subplot(2, 2, 3)
            # ax3.plot(pb._design_default_filter(shift))
            #
            # plt.show()
            
    def test_frac_time_shift_with_random(self):
        """ Test frac_time_shift with a random signal"""
        
        N = 1024
        p,q = (7, 6) # resampling parameters
        b, a = scipy.signal.butter(10,.25)
        shifts = (0.5, 1/6, numpy.random.uniform(low=-5, high=+5))
        
        for shift in shifts:
            
            noise = numpy.pad(numpy.random.rand(N//2), (N//4, N//4))
            filt_noise = scipy.signal.filtfilt(b, a, noise)
            
            res1 = pb.frac_time_shift(scipy.signal.resample_poly(filt_noise, p, q), p/q * shift)
            res2 = scipy.signal.resample_poly(pb.frac_time_shift(filt_noise, shift), p, q)
            error = numpy.max(numpy.abs(res1-res2))

            # ax1 = plt.subplot(2, 2, 1)
            # ax1.plot(res1, label='resamp/shift')
            # ax1.plot(res2, '.', label='shift/resamp')
            #
            # ax1.legend()
            # ax1.grid()
            # 
            # ax2 = plt.subplot(2, 2, 2)
            # ax2.plot(numpy.abs(res1-res2))
            # ax2.grid()
            # plt.show()
            
            self.assertLess(error, 10**pb.LOG10_REJECTION)

    def test_delayseq_time_shift_with_sinus(self):
        """ Test delayseq with sinusoidal signal"""

        N = 1024
        shifts = (0.5, 1/6, numpy.random.uniform(low=-5, high=+5))
        time = numpy.arange(-N/2,N/2)
        sigma = N/4

        for shift in shifts:

            time_shifted = time - shift
            error = []

            for f in numpy.arange((1-pb.DELAYSEQ_ROLL_OFF) * N)/(2*N):
                phi =  2 * math.pi * numpy.random.rand()
                signal = sinegauss(time, sigma, f, phi)
                estimate_shifted = pb.delayseq(signal, shift)
                exact_shifted = sinegauss(time_shifted, sigma, f, phi)
                error = numpy.max(numpy.abs(estimate_shifted-exact_shifted))
                # error.append(numpy.max(numpy.abs(estimate_shifted-exact_shifted)))
                self.assertTrue(numpy.isclose(error, 0))

            # ax1 = plt.subplot(2, 2, 1)
            # ax1.semilogy(error)
            # ax1.grid()
            #
            # ax2 = plt.subplot(2, 2, 2)
            # ax2.plot(signal, label='orig')
            # ax2.plot(estimate_shifted, label='shifted')
            # ax2.plot(exact_shifted, label='exact')
            # ax2.grid()
            # ax2.plot(numpy.abs(estimate_shifted-exact_shifted), label='error')
            # ax2.legend()
            #
            # plt.show()
            
    def test_delayseq_with_random(self):
        """ Test delayseq with a random signal"""
        
        N = 1024
        p,q = (7, 6) # resampling parameters
        b, a = scipy.signal.butter(10,.25)
        shifts = (0.5, 1/6, numpy.random.uniform(low=-5, high=+5))
        
        for shift in shifts:
            
            noise = numpy.pad(numpy.random.rand(N//2), (N//4, N//4))
            filt_noise = scipy.signal.filtfilt(b, a, noise)
            
            res1 = pb.delayseq(scipy.signal.resample_poly(filt_noise, p, q), p/q * shift)
            res2 = scipy.signal.resample_poly(pb.delayseq(filt_noise, shift), p, q)
            error = numpy.max(numpy.abs(res1-res2))

            # ax1 = plt.subplot(2, 2, 1)
            # ax1.plot(res1, label='resamp/shift')
            # ax1.plot(res2, '.', label='shift/resamp')
            #
            # ax1.legend()
            # ax1.grid()
            #
            # ax2 = plt.subplot(2, 2, 2)
            # ax2.plot(numpy.abs(res1-res2))
            # ax2.grid()
            # plt.show()
            
            self.assertLess(error, 10**pb.DELAYSEQ_LOG10_REJECTION)

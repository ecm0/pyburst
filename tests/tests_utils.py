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

    def test_angle_between(self):
        """ Test of angle between vectors """

        tests = [([1,0,0], [1,0,0], 0), \
                   ([1,0,0], [0,1,0], math.pi/2), \
                   ([1,0,0], [0,0,1], math.pi/2)]
        for v1, v2, res in tests:
            self.assertEqual(pb.angle_between(v1, v2), res)
        
    def test_frac_time_shift_output_size(self):
        """ Test frac_time_shift: check input and output sizes match"""

        N = 1024
        shift = numpy.random.uniform(low=-5, high=+5)
        shifted = pb.frac_time_shift(numpy.zeros(N), shift)
        self.assertEqual(shifted.size, N)
        
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

    def test_delayseq_output_size(self):
        """ Test delayseq: check input and output sizes match"""

        N = 1024
        shift = numpy.random.uniform(low=-5, high=+5)
        shifted = pb.delayseq(numpy.zeros(N), shift)
        self.assertEqual(shifted.size, N)

    def test_delayseq_output_size_other(self):
        """ Test delayseq: check input and output sizes match"""

        N = 8256
        shift = numpy.random.uniform(low=70, high=80)
        shifted = pb.delayseq(numpy.zeros(N), -shift)
        self.assertEqual(shifted.size, N)
        
    def test_delayseq_with_sinus(self):
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

    def test_orthonormalize(self):
        """ Test Gram-Schmidt orthonormalization of two vectors"""

        u1 = numpy.array([1,2,2])
        u2 = numpy.array([-1,0,2])
        v, _ = pb.orthonormalize(u1, u2)
        
        # Excepted result is:
        r1 = numpy.array([-1,-2,-2])/3
        r2 = numpy.array([2,1,-2])/3

        self.assertTrue(numpy.allclose(v[0], r1))
        self.assertTrue(numpy.allclose(v[1], r2))

    def test_orthonormalize_dominant(self):
        """ Test dominant polarization frame orthormalization
        in two simple cases
        """
        
        u1 = numpy.array([1,+0.5,0])
        u2 = numpy.array([1,-0.5,0])
        v, _ = pb.orthonormalize(u1, u2, dominant_polar_frame=True)
        print(v)
        
        # Excepted result is:
        r1 = numpy.array([1,0,0])
        r2 = numpy.array([0,1,0])
        print(r1, r2)

        print(numpy.cross(v[0], r1))
        print(numpy.cross(v[1], r2))
        
        self.assertTrue(numpy.allclose(numpy.cross(v[0], r1), 0))
        self.assertTrue(numpy.allclose(numpy.cross(v[1], r2), 0))

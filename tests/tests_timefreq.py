from unittest import TestCase

import math
import numpy
import pyburst.timefreq as tf

import matplotlib.pyplot as plt

def sinegauss(time, sigma, f, phi):
    return numpy.exp(-time**2/(2*sigma)) * \
        numpy.sin(2 * math.pi * f * time + phi)

class TestUtils(TestCase): 

    def test_transform(self):
        """Test forward transform"""

        N = 1024
        signal = numpy.zeros(N)
        tfrep = tf.TimeFreqTransform("dgt", "hanning", 64, 512)
        tfmap = tfrep.forward(signal)
        self.assertTrue(numpy.all(tfmap.data==0))
        
#     def test_inversion_ok(self):
#        """Test forward/backward transform inversion"""
        
#    def test_timeshift_output_size(self):
#        """ Test frac_time_shift: check input and output sizes match"""
#
#        N = 1024
#        self.assertEqual(shifted.size, N)
        
    # def test_timeshift_with_sinus(self):
    #     """ Test transform timeshifting using a sinusoidal signal"""

    #     N = 1024
    #     shifts = (0.5, 1/6, numpy.random.uniform(low=-5, high=+5))
    #     time = numpy.arange(-N/2,N/2)
    #     sigma = N/4

    #     for shift in shifts:

    #         time_shifted = time - shift
    #         # error = []

    #         for f in numpy.arange(math.floor((pb.STOPBAND_CUTOFF_F-pb.ROLL_OFF_WIDTH) * N))/N:
    #             phi =  2 * math.pi * numpy.random.rand()
    #             signal = sinegauss(time, sigma, f, phi)
    #             estimate_shifted = pb.frac_time_shift(signal, shift)
    #             exact_shifted = sinegauss(time_shifted, sigma, f, phi)
    #             error = numpy.max(numpy.abs(estimate_shifted-exact_shifted))
    #             # error.append(numpy.max(numpy.abs(estimate_shifted-exact_shifted)))
    #             self.assertLess(error, 10**pb.LOG10_REJECTION)
                
    #         # ax1 = plt.subplot(2, 2, 1)
    #         # ax1.semilogy(error)
    #         # ax1.grid()
    #         #
    #         # ax2 = plt.subplot(2, 2, 2)
    #         # plt.plot(signal, label='orig')
    #         # ax2.plot(estimate_shifted, label='shift')
    #         # ax2.plot(exact_shifted, '.', label='exact')
    #         # ax2.grid()
    #         # plt.plot(numpy.abs(estimate_shifted-exact_shifted), label='error')
    #         # ax2.legend()
    #         #
    #         # ax3 = plt.subplot(2, 2, 3)
    #         # ax3.plot(pb._design_default_filter(shift))
    #         #
    #         # plt.show()
            

from unittest import TestCase

import math
import numpy
import pyburst.timefreq as tf

import matplotlib.pyplot as plt

def sinegauss(time, sigma, f, phi):
    return numpy.exp(-time**2/(2*sigma)) * \
        numpy.sin(2 * math.pi * f * time + phi)

class TestUtils(TestCase): 

    def test_transform_dgt(self):
        """Test dgt transform"""

        N = 1024
        signal = numpy.zeros(N)
        tfrep = tf.TimeFreqTransform("dgt", "hanning", 64, 512)
        tfmap = tfrep.forward(signal)
        rec = tfrep.backward(tfmap.data)

        self.assertEqual(tfmap.data.shape[0], 512)
        self.assertEqual(tfmap.data.shape[1], N/64)
        self.assertTrue(numpy.all(rec==0))

    def test_transform_dgtreal(self):
        """Test dgtreal transform"""

        N = 1024
        signal = numpy.zeros(N)
        tfrep = tf.TimeFreqTransform("dgtreal", "hanning", 64, 512)
        tfmap = tfrep.forward(signal)
        rec = tfrep.backward(tfmap.data)
        
        self.assertEqual(tfmap.data.shape[0], 257)
        self.assertEqual(tfmap.data.shape[1], N/64)
        self.assertTrue(numpy.all(rec==0))
        
    def test_inversion(self):
        """Test forward/backward transform inversion"""
        
        N = 1024
        signal = numpy.random.normal(0, 1, N)
        tfrep = tf.TimeFreqTransform("dgtreal", "hanning", 64, 512)
        tfmap = tfrep.forward(signal)
        rec = tfrep.backward(tfmap.data)
        
        self.assertTrue(numpy.allclose(numpy.abs(signal-rec), 0))

    # def test_timeshift_with_sinus(self):
    #     """ Test transform timeshifting using a sinusoidal signal"""

    #     N = 1024
    #     shifts = (0.0,) # (0.0, 0.5, 1/6, numpy.random.uniform(low=-5, high=+5))
    #     time = numpy.arange(-N/2,N/2)
    #     sigma = N/4
        
    #     shift = shifts[0]
    #     # for shift in shifts:
        
    #     time_shifted = time - shift
    #     # error = []

    #     f = numpy.random.rand()/2
    #     phi =  2 * math.pi * numpy.random.rand()
    #     signal = sinegauss(time, sigma, f, phi)
    #     exact_shifted = sinegauss(time_shifted, sigma, f, phi)

    #     tfrep = tf.TimeFreqTransform("dgtreal", "gauss", 64, 512)
    #     kernel, window = tfrep.reduced_kernel(shift)
    #     tfmap = tfrep.forward(signal)
    #     tfmap_shifted = tfmap.timeshift(shift)
        
    #     tfmap_exact_shifted = tfrep.forward(exact_shifted)
        
    #     error = numpy.max(numpy.abs(tfmap_shifted.data-tfmap_exact_shifted.data))
        
        # ax1 = plt.subplot(2, 2, 1)
        # plt.plot(signal, label='orig')
        # ax1.plot(exact_shifted, '.', label='exact')
        # ax1.grid()
        # ax1.legend()
        # plt.subplot(2, 3, 1)
        # # plt.plot(window)
        # # plt.pcolormesh(numpy.real(kernel))
        # plt.pcolormesh(numpy.real(tfmap_exact_shifted.data))
        # plt.grid()
        # plt.title('exact - real')
        # plt.colorbar()
        
        # plt.subplot(2, 3, 2)
        # plt.pcolormesh(numpy.imag(tfmap_exact_shifted.data))
        # plt.grid()
        # plt.title('exact - imag')
        # plt.colorbar()
        
        # plt.subplot(2, 3, 3)
        # plt.pcolormesh(numpy.abs(tfmap_exact_shifted.data))
        # plt.grid()
        # plt.title('exact - abs')
        # plt.colorbar()
        
        # plt.subplot(2, 3, 4)
        # plt.pcolormesh(numpy.real(tfmap_shifted.data))
        # plt.grid()
        # plt.title('approx - real')
        # plt.colorbar()

        # plt.subplot(2, 3, 5)
        # plt.pcolormesh(numpy.imag(tfmap_shifted.data))
        # plt.grid()
        # plt.title('approx - imag')
        # plt.colorbar()
        
        # plt.subplot(2, 3, 6)
        # plt.pcolormesh(numpy.abs(tfmap_shifted.data))
        # plt.grid()
        # plt.title('approx - abs')
        # plt.colorbar()

        # plt.show()
         

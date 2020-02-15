import math
import numpy
from scipy.signal import convolve2d
import logging

from ltfatpy import dgt, idgt, dgtreal, idgtreal, gabwin
from ltfatpy.gabor.gabdual import gabdual

from . import utils

class TimeFreqTransform(object):
    """A TimeFreqTransform object characterizes a time-frequency transformation
    of time series.
    """
    
    def __init__(self, transform, window_type, time_step, num_freqs):
        """
        transform -- type of the time-frequency transform
        window_type -- type of the analysis/synthesis window (string)
        time_step -- size of the time step
        num_freqs -- number of frequencies

        Note: the direct window is used for the synthesis, and the dual 
        window is used for the analysis. The reconstructed signal can be
        written as the sum of modulated windows.
        """
        
        self.transform = transform
        self.window_type = window_type
        self.analysis_window = {'name' : ('dual', window_type), 'M' : num_freqs}
        self.synthesis_window = {'name': window_type, 'M': num_freqs}
        self.time_step = time_step
        self.num_freqs = num_freqs
        
        # allowed_keys = ['strip_edges']
        # for k, v in kwargs.iteritems() if k in allowed_keys:
        #    setattr(self, k, v)
       
    def forward(self, data):
        """
        Forward transform
        data -- input data (Numpy array)
        """
            
        if self.transform == 'dgtreal':
            out,_,_ = dgtreal(data, self.analysis_window, \
                              self.time_step, self.num_freqs)
            return TimeFreq(self, out, data.size)
        elif self.transform == 'dgt':
            out,_,_ = dgt(data, self.analysis_window, \
                              self.time_step, self.num_freqs)
            return TimeFreq(self, out, data.size)         
        # elif test expression:
        else: 
            logging.warning("Unknown transform")
            return None

    def backward(self, data):
        """
        Backward transform
        data -- input data (Numpy array)
        """
            
        if self.transform == 'dgtreal':
            out,_ = idgtreal(data, self.synthesis_window, \
                            self.time_step, self.num_freqs)
            return out
        elif self.transform == 'dgt':
            out,_ = idgt(data, self.synthesis_window, \
                              self.time_step, self.num_freqs)
            return out
        # elif test expression:
        else: 
            logging.warning("Unknown transform")
            return None

    def reduced_kernel(self, delay=0):
        """
        Returns a reduced form of the reproducing kernel, with
        optionally a (possibly fractional) time shift.
        The kernel is reduced to its components at freq=0.
        The anticipated relative error due to this reduction 
        is also returned.
        """
        # assert delay < dualwin.size

        synth_window,_ = gabwin(self.synthesis_window, \
                                self.time_step, \
                                self.num_freqs, \
                                512)
        window = numpy.pad(numpy.fft.fftshift(synth_window), (512))
        tfmap = self.forward(utils.delayseq(window, \
                                      delay))
        # normalized_map = numpy.fft.fftshift(tfmap.data, axes=0)/numpy.linalg.norm(tfmap.data)
        kernel = numpy.vstack((tfmap.data[2:0:-1,7:18],tfmap.data[0:3,7:18]))
        
        return kernel, window
        # return normalized_map[0,:], numpy.sum(normalized_map[1:,:])

class TimeFreq(object):
    """A TimeFreq object characterizes the time-frequency map 
    produced by a given transformation.
    """
    
    def __init__(self, transform, data, original_size):
        """
        transform -- time-frequency transform (TimeFreqTransform) 
        used to produce 
        data -- 2D Numpy array
        original_size -- size of the original time-series
        """
        
        self.transform = transform
        self.data = data
        self.original_size = original_size
        
        # allowed_keys = ['strip_edges']
        # for k, v in kwargs.iteritems() if k in allowed_keys:
        #    setattr(self, k, v)
        
    def freqs(self, sampling_freq=1):
        """
        Returns the frequency axis of the time-frequency map
        """
        return sampling_freq * numpy.arange(self.data.shape[0]) \
            /(2 * (self.data.shape[0]-1))
 
    def freq_index(self, freq, sampling_freq=1):
        """
        Compute the row index associate to a physical frequency
        """
        assert freq >=0 and freq <= sampling_freq/2, 'frequency is not in Nyquist band'
        return  int(numpy.round(2 * self.data.shape[0] * freq/sampling_freq))

    def times(self, sampling_freq=1):
        """
        Returns the time axis of the time-frequency map
        """
        return self.transform.time_step/sampling_freq * numpy.arange(self.data.shape[1])

    def trim_edges(self, size, value=None):
        """ 
        Strip map edges or set them to a given value
        """
        assert self.data.shape[1] > 2 * size, 'Array does not have enough columns'

        if value == None:
            self.data = self.data[:,size:self.data.shape[1]-size]
            self.original_size -= 2 * size * self.transform.time_step
        else:
            self.data[:, 0:size] = value
            self.data[:, -size:] = value
        return self
        
    def highpass(self, cutoff_freq, sampling_freq=1):
        """ 
        Set frequencies below cut-off to zero
        """
        self.data[0:self.freq_index(cutoff_freq, sampling_freq)] = 0
        return self

    def timeshift(self, delay):
        """
        Shift the time-frequency map by a (possibly fractional) number of samples
        by an approximate kernel-based interpolation method.
        """
        
        # kernel, error_estimate = self.transform.reduced_kernel(delay)
        # print(kernel.shape)

        # phases = numpy.exp(-2j * math.pi * delay * self.freqs())
        # tfmap = phases[:, numpy.newaxis] * \
        #        numpy.apply_along_axis(numpy.convolve, 1, \
        #                               self.data, kernel, \
        #                               'same')
        # tfmap = phases[:, numpy.newaxis] * \
        #                     convolve2d(self.data, kernel, \
        #                                mode='same')
        
        # return TimeFreq(self.transform, tfmap, self.original_size)
        return None
        
    def marginal(self, axis='freq', method='medianmean'):
        """
        Marginalize time-freq map in time or freq using
        the selected method ('medianmean' [default], 'mean' or 'median')

        Note: the bias of the median estimator for the mean is described in
        arXiv:gr-qc/0509116 appendix B for details.
        """

        summed_axis_ix = 0 if axis=='time' else 1 if axis=='freq' else None

        ## XXX The bias of median estimator appears to be smaller than the one
        ## XXX calculated by _median_bias() ! This correction should possibility
        ## XXX be applied to the PSD and not to the ASD?
        
        if method == 'mean':
            # return numpy.linalg.norm(self.data, axis=summed_axis_ix)
            # return numpy.sqrt(numpy.mean(numpy.abs(self.data)**2, axis=summed_axis_ix))
            return numpy.mean(numpy.abs(self.data), axis=summed_axis_ix)
        elif method == 'median':
            return numpy.median(numpy.abs(self.data), axis=summed_axis_ix)
                # / _median_bias(self.data.shape[summed_axis_ix])
                # return numpy.sqrt(numpy.median(numpy.abs(self.data)**2, axis=summed_axis_ix))
        elif method == 'medianmean':
            medians = []
            for offset in range(2):
                # Select even and odd rows/columns
                data = numpy.take(self.data, range(offset,self.data.shape[summed_axis_ix],2), axis=summed_axis_ix)
                # Compute median separately on even and odd rows/columns ...
                medians.append(numpy.median(numpy.abs(data), axis=summed_axis_ix))
                # / _median_bias(data.shape[summed_axis_ix]))
                # ... and average
            return numpy.squeeze(numpy.mean(numpy.array(medians), axis=0))
        else:
            logging.warning("Unknown averaging method")
            return None
        
def _median_bias(n):
    """
    Compute the bias of median estimates vs the data size
    arXiv:gr-qc/0509116 appendix B for details.
    """

    if n >= 1000:
        return numpy.log(2)
    ans = 1
    for i in range(1, int((n - 1) / 2 + 1)):
        ans += 1.0 / (2*i + 1) - 1.0 / (2*i)
        
    return ans
       
def _col2diag(a):
    """
    Transforms columns to diagonals. This function transforms the first column 
    to the main diagonal. The second column to the first side-diagonal below the 
    main diagonal and so on. 
 
    This is the Python implementation of comp_col2diag.m function in LTFAT.
    """
    
    b = numpy.zeros_like(a)
    for n,col in enumerate(a.T):
        numpy.fill_diagonal(b[n:],col[n:])
        if n > 0:
            numpy.fill_diagonal(b[:,a.shape[0]-n:],col[:n])            
    return b

def twisted_conv(f, g):
    """
    Twisted convolution. Computes the twisted convolution of the square matrices
    f and g.

    Let h=twisted_conv(f,g) for f,g being L x L matrices. Then h is given by

                L-1 L-1
      h(m,n) =  sum sum f(k,l) * g(m-k,n-l)*exp(-2*pi*i*(m-k)*l/L);
                l=0 k=0

    where m-k and n-l are computed modulo L.

    This is the Python implementation of tconv.m function in LTFAT.
    """

    assert f.shape == g.shape
    
    a = _col2diag(numpy.fft.ifft(f, axis=0)) * f.shape[0]
    b = _col2diag(numpy.fft.ifft(g, axis=0)) * g.shape[0]
    return numpy.fft.fft(comp_col2diag(numpy.dot(a, b)), axis=0) / f.shape[0]

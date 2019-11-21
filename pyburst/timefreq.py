import math
import numpy

from ltfatpy import dgtreal, idgtreal

class TimeFreqTransform(object):
    """A TimeFreqTransform object characterizes a time-frequency transformation
    of time series.
    """
    
    def __init__(self, transform, window_type, time_step, num_freqs):
        """
        transform -- type of the time-frequency transform
        window_type -- type of the analysis window (string)
        time_step -- size of the time step
        num_freqs -- number of frequencies
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
            return idgtreal(data, self.synthesis_window, \
                            self.time_step, self.num_freqs)
        # elif test expression:
        else: 
            logging.warning("Unknown transform")
            return None

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
        return sampling_freq * numpy.arange(self.data.shape[0])/(2 * (self.data.shape[0]-1))
 
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
       

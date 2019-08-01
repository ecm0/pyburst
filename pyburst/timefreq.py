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
        return sampling_freq * numpy.arange(self.data.shape[0])/(2 * self.data.shape[0])
 
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

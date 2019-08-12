import math
import numpy

ALLOWED_INVERSION_METHODS = ('narrowband-approx', 'timefreq-shift')

def mixing(method):
    """ Compute mixing matrix at each sky position and frequencies

    Shapes of mixing matrices

    Inputs:
    * delays: Ndet x Nsky
    * antenna_patterns: Ndet x Npol x Nsky
    * asds: Ndet x Nfreq
    * freqs: Ndet x Nfreq

    Outputs:
    * mixing_whitened: Nsky x Nfreq x Ndet x Npol
    * mixing_whitened_inverse: Nsky x Nfreq x Npol x Ndet
    """

    # Indices:
    # 
    # * s: index of sky pixel
    # * d: detector index
    # * p: polarization index (plus, cross)
    # * f: frequency index
    # 
    # Note: this can be done only once for an entire segment

    if method == 'narrowband-approx':    

        # XXX Is the sign in the phase correct? XXX
        phases = numpy.exp(-2*numpy.pi*1j* numpy.einsum('ds,df->sdf', delays, freqs))
        mixing = numpy.einsum('dps,sdf->sfdp', antenna_patterns, phases)
        mixing_whitened = numpy.einsum('dps,df,sdf->sfdp', antenna_patterns, 1/asds, phases)
        mixing_whitened_inverse = numpy.linalg.pinv(mixing_whitened)
        return mixing_whitened, mixing_whitened_inverse
    
    elif method == '':
    else:
        logging.warning("Unknown transform")
        return None

class TimeFreqCluster(object):
    """A TimeFreqCluster represents a cluster of time-frequency pixels.
    """

    def __init__:
    
    def selection():

    def inverse(self, method):
        """
        """

        signal_estimate_tfmaps = numpy.einsum('sfpd,dft->sftp', \
                                            mixing_whitened_inverse[:,idx_freq, ...], \
                                            reduced_tfmaps)
        response_estimate_tfmaps = numpy.einsum('sfdp,sftp->sdft', \
                                            mixing_whitened[:,idx_freq, ...], \
                                            signal_estimate_tfmaps)
        estimation_error = numpy.linalg.norm(reduced_tfmaps[numpy.newaxis,:] \
                                             - response_estimate_tfmaps, \
                                             axis=(-2,-1)).sum(axis=-1)
        skymap = sky.feed(estimation_error)
        
    def reconstruction():
        """
        """

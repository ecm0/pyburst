import math
import numpy

ALLOWED_INVERSION_METHODS = ('narrowband-approx', 'timefreq-shift', 'time-shift')

class TimeFreqCluster(object):
    """A TimeFreqCluster represents a cluster of time-frequency pixels.
    """

    def __init__:
        pixel
        times
        freqs
        
    def selection():

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
        phases = numpy.exp(-2*numpy.pi*1j* numpy.einsum('ds,df->sdf', delays, freqs))
        mixing = numpy.einsum('dps,sdf->sfdp', antenna_patterns, phases)
        mixing_whitened = numpy.einsum('dps,df,sdf->sfdp', antenna_patterns, 1/asds, phases)
        mixing_whitened_inverse = numpy.linalg.pinv(mixing_whitened)
        return mixing_whitened, mixing_whitened_inverse
    elif method == 'timefreq-shift':
        # phases = np.exp(-2*np.pi*1j* np.einsum('ds,df->sdf', delays, freqs))
        # mixing = np.einsum('dps,sdf->sfdp', antenna_patterns, phases)
        mixing_whitened = np.einsum('dps,df->sfdp', antenna_patterns, 1/asds)
        mixing_whitened_inverse = np.linalg.pinv(mixing_whitened)
        return mixing_whitened, mixing_whitened_inverse
        
    else:
        logging.warning("Unknown transform")
        return None

    def inverse(self, method):
        """
        """

        if method == 'narrowband-approx':    
            signal_estimate_tfmaps = numpy.einsum('sfpd,dft->sftp', \
                                                  mixing_whitened_inverse[:,idx_freq, ...], \
                                                  reduced_tfmaps)
            response_estimate_tfmaps = numpy.einsum('sfdp,sftp->sdft', \
                                                    mixing_whitened[:,idx_freq, ...], \
                                                    signal_estimate_tfmaps)
            estimation_error = numpy.linalg.norm(reduced_tfmaps[numpy.newaxis,:] \
                                                 - response_estimate_tfmaps, \
                                                 axis=(-2,-1)).sum(axis=-1)
        elif method == 'timefreq-shift':
            # I here shift the reduced tf maps and calculate the estimation error for each sky point
            # First the time shift in the t-f domain    
            reduced_shift_maps = np.zeros([3,len(idx_freq),len(idx_time)+12], dtype=np.complex_)
            for this_f in np.arange(len(idx_freq)):
                for this_det in np.arange(3):
                    temp1 = delayfunc.delayseq(Kernel1,8*delays[this_det,sky_index],1)
                    temp2 = spsig.convolve(reduced_tfmaps[this_det,this_f,:],temp1)
                    temp3 = np.roll(temp2,-6)
                    reduced_shift_maps[this_det,this_f,:] = np.array(temp3,dtype=np.complex_)
            
            signal_estimate_tfmaps_one = np.einsum('fpd,dft->ftp', \
                                    np.squeeze(mixing_whitened_inverse[sky_index,idx_freq, ...]), \
                                    reduced_tfmaps)
            response_estimate_tfmaps_one = np.einsum('fdp,ftp->dft', \
                                    np.squeeze(mixing_whitened[sky_index,idx_freq, ...]), \
                                    signal_estimate_tfmaps_one)
            estimation_error_conv[sky_index] = np.linalg.norm(reduced_tfmaps - \
                                response_estimate_tfmaps_one, \
                                axis = (-3,-1)).sum()
        else:
            logging.warning("Unknown transform")
            return None

        skymap = sky.feed(estimation_error)
        
    def reconstruction():
        """
        """

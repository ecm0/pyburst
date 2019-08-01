import numpy

import lal
from lal import  LALDetectorIndexLHODIFF,LALDetectorIndexLLODIFF, \
                 LALDetectorIndexVIRGODIFF, LALDetectorIndexKAGRADIFF, \
                 LALDetectorIndexLIODIFF

DETECTOR_SITES = {
    'H1': LALDetectorIndexLHODIFF,
    'L1': LALDetectorIndexLLODIFF,
    'V1': LALDetectorIndexVIRGODIFF,
    'K1': LALDetectorIndexKAGRADIFF,
    'I1': LALDetectorIndexLIODIFF
    }

import lalsimulation
from lalsimulation import SimDetectorStrainREAL8TimeSeries

import gwpy
from gwpy.timeseries import TimeSeries

class Detector(object):
    """
    A Detector object characterises a gravitational wave (GW) interferometric detector
    """

    def __init__(self, detector):
        """
        detector  -- label string of the detector
        descriptor -- LAL descriptor
        location -- geographic location of the detector
        response -- response matrix

        """
        self.name = detector
        self.descriptor =  lal.CachedDetectors[DETECTOR_SITES[detector]]
        self.location = lalsimulation.DetectorPrefixToLALDetector(detector).location
        self.response = lalsimulation.DetectorPrefixToLALDetector(detector).response
        
    def __str__(self):
        return self.name
            
    def antenna_pattern(self, ra, dec, psi, ref_time=None):
        """ Compute antenna response
            ra or long: right ascension or longitude in radians if ref_time is None 
            dec or lat: declination or latitude in radians if ref_time is None
            ref_time: reference time used to compute equatorial sky coordinates
        """
        
        ra = numpy.atleast_1d(ra) # change to vector if scalar
        dec = numpy.atleast_1d(dec) # change to vector if scalar
                
        assert ra.size == dec.size, "RA and dec arrays don't have the same size"
        
        gmst_rad = lal.GreenwichMeanSiderealTime(ref_time) if ref_time is not None else 0.0
                               
        fplus = []
        fcross = []
        for (ra_val, dec_val) in zip(ra, dec):
            fplus_val,fcross_val = lal.ComputeDetAMResponse(self.descriptor.response, \
                                                       ra_val, dec_val, psi, gmst_rad)
            fplus.append(fplus_val)
            fcross.append(fcross_val)
            
        return numpy.squeeze(numpy.array(fplus)), numpy.squeeze(numpy.array(fcross))
    
    def project_strain(self, hplus, hcross, time, ra, dec, psi):
        """ Project hplus and hcross onto the detector frame 
        assuming a given sky location and polarization of the source.
        Apply consistent time delay due to wave propagation.
        """

        assert hplus.data.length == hcross.data.length
        assert hplus.deltaT == hcross.deltaT
                        
        return TimeSeries.from_lal(SimDetectorStrainREAL8TimeSeries( \
                                        hplus, hcross, \
                                        ra, dec, psi, \
                                        self.descriptor))

    def time_delay_from_earth_center(self, ra, dec, ref_time=None):
        """ Returns the time delay from the earth center
            ra or long: right ascension or longitude in radians if ref_time is None 
            dec or lat: declination or latitude in radians if ref_time is None
        """
        
        ra = numpy.atleast_1d(ra) # change to vector if scalar
        dec = numpy.atleast_1d(dec) # change to vector if scalar
                
        assert ra.size == dec.size, "RA and dec arrays don't have the same size"
        
        time = ref_time if ref_time is not None else 0.0

        delays = []
        for (ra_val, dec_val) in zip(ra, dec):
            delays.append(lal.TimeDelayFromEarthCenter(self.location, ra_val, dec_val, time))
    
        return numpy.squeeze(numpy.array(delays))

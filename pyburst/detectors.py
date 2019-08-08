import logging
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

# Reference date where Greenwich Mean Sidereal Time is 0 hr
# lal.GreenwichMeanSiderealTime(REFDATE_GMST_ZERO) results in -9.524206245228903e-15
# At this specific time, the equatorial coordinate system coincides with the geographic
# coordinate system to precision machine
REFDATE_GMST_ZERO = lal.LIGOTimeGPS(630696086, 238589290) # Dec 31 1999, 17:21:13 238589

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
            
    def antenna_pattern(self, skypoints, ref_time=None, psi=0):
        """ Compute antenna response
            skypoints: Skypoint object or list of Skypoint objects
            ref_time: reference time when using the equatorial coordinate system
            psi: optional polarization angle (default: 0)
        """

        if not isinstance(skypoints, list):
            skypoints = [skypoints]

        skypoints[0].coordsystem.is_valid()

        if skypoints[0].coordsystem.name == 'equatorial':
            assert ref_time is not None, 'Reference time is required when using the equatorial coordinate system'
        
        gmst_rad = lal.GreenwichMeanSiderealTime(ref_time) if ref_time is not None else 0.0
                               
        f = [lal.ComputeDetAMResponse(self.descriptor.response, \
                                      *p.coords(fmt='lonlat',unit='radians'), \
                                      psi, gmst_rad) for p in skypoints]

        f = numpy.squeeze(numpy.array(f))

        if f.ndim == 1:
            return tuple(f) # fplus, fcross
        else:
            return f[:,0], f[:,1] # fplus, fcross
    
    def project_strain(self, hplus, hcross, time, ra, dec, psi):
        """ Project hplus and hcross onto the detector frame 
        assuming a given sky location and polarization of the source.
        Apply consistent time delay due to wave propagation.
        
        hplus, hcross: plus and cross polarisation (REAL8TimeSeries)
        ra: right ascension in radians
        dec: declination in radians
        psi: polarization angle in radians

        """

        assert hplus.data.length == hcross.data.length
        assert hplus.deltaT == hcross.deltaT
                        
        return TimeSeries.from_lal(SimDetectorStrainREAL8TimeSeries( \
                                        hplus, hcross, \
                                        ra, dec, psi, \
                                        self.descriptor))

    def time_delay_from_earth_center(self, skypoints, ref_time=None):
        """ Returns the time delay from the earth center
            skypoints: Skypoint object or list of Skypoint objects
            ref_time: reference time when using the equatorial coordinate system
        """

        if not isinstance(skypoints, list):
            skypoints = [skypoints]

        skypoints[0].coordsystem.is_valid()

        if skypoints[0].coordsystem.name == 'equatorial':
            assert ref_time is not None, 'Reference time is required when using the equatorial coordinate system'
        elif skypoints[0].coordsystem.name == 'geographic':
            # lal.TimeDelayFromEarthCenter() only takes equatorial coordinates.
            # Equatorial coordinates at time=REFDATE_GMST_ZERO coincide with
            # geographic coordinates
            ref_time = REFDATE_GMST_ZERO
        else:
            logging.warning('Unknown format')
            return None
        
        delays = [lal.TimeDelayFromEarthCenter(self.location, \
                                                   *p.coords(fmt='lonlat',unit='radians'), \
                                                   ref_time) for p in skypoints]
    
        return numpy.squeeze(numpy.array(delays))

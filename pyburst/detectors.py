import math
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

import lalsimulation
from lalsimulation import SimDetectorStrainREAL8TimeSeries

import gwpy
from gwpy.timeseries import TimeSeries

from .skymaps import Coordsystem
from .skymaps import Skypoint

# Reference date where Greenwich Mean Sidereal Time is 0 hr
# lal.GreenwichMeanSiderealTime(REFDATE_GMST_ZERO) results in -9.524206245228903e-15
REFDATE_GMST_ZERO = lal.LIGOTimeGPS(630696086, 238589290) # Dec 31 1999, 17:21:13 238589

# Fiducial equatorial coordinate system chosen with a reference time
# equal to REFDATE_GMST_ZERO. At this specific time, the equatorial
# coordinate system coincides with the geographic coordinate system
# to precision machine
FIDUCIAL_EQUATORIAL_COORDSYS_GMST_ZERO = Coordsystem('equatorial', \
                                                     ref_time=REFDATE_GMST_ZERO)

class Network(object):
    """
    A Network object represents an array of gravitational wave 
    (GW) interferometric detector.
    """

    def __init__(self, detectors):
        """
        detectors  -- list of detector labels (str)
        """
        self.detectors = [Detector(d) for d in detectors]
        
    def __str__(self):
        return " ".join([d.name for d in self.detectors])

    def plane(self, coordsystem, indices=[0, 1, 2], npoints=50):
        """
        Returns a list of Skypoint objects pointing in the direction
        of the plane for by three detectors.

        coordsystem: Coordsystem object that describes the 
        indices: list of indices that selects the 3 reference detectors (default: [0, 1, 2])
        npoints: number of skypoints (default: 50)
        """

        assert len(self.detectors) > 2, "Not enough detectors to define a plane"

        detectors = [self.detectors[i] for i in indices]
        
        plane = numpy.array([detectors[2].location-detectors[0].location, \
                                 detectors[1].location-detectors[0].location])
        plane_basis, _, _ = numpy.linalg.svd(plane.T)
        
        pts = []
        for az in np.linspace(0, 2*math.pi, npoints):
            pts.append(Skypoint.from_cart(plane_basis[:,0] * math.cos(az) +  \
                                            plane_basis[:,1] * math.sin(az), \
                                            coordsystem))
        return pts

    def antenna_pattern(self, skypoints, time=None, psi=0, dominant_frame=False):
        """
        Compute antenna response matrix for all detector in the network 
        and for a list of skypoints
        
        skypoints: Skypoint object or list of Skypoint objects
        time: time date when to compute the antenna pattern (default: None) 
        psi: optional polarization angle (default: 0)
        """

        res = np.array([d.antenna_pattern(skypoints, time, psi) \
                        for d in network.detectors])
        
        if dominant_frame:
            return orthonormalize(res[:,0], res[:,1], \
                               dominant_polar_frame=True)
        else:
            return res

        
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
            
    def antenna_pattern(self, skypoints, time=None, psi=0):
        """ Compute antenna response of a detector for a list of skypoints

            skypoints: Skypoint object or list of Skypoint objects
            time: time date when to compute the antenna pattern (default: None) 
            psi: optional polarization angle (default: 0)

            This function uses XLALComputeDetAMResponse() that takes equatorial 
            coordinates (RA and dec) and the Greenwich Mean Sidereal Time (GMST) 
            to define the sky location.

            If the skypoints are given in the equatorial coordinate system, 
            the antenna pattern is computed at the provided time if not None, otherwise
            it is computed at the reference time of the coordinate system of 
            the skypoints.

            If the skypoints are given in the geographic coordinate system,
            they are mapped to fiducial equatorial coordinate system with 
            GMST = 0 hr (REFDATE_GMST_ZERO).
        """

        if not isinstance(skypoints, list):
            skypoints = [skypoints]
            
        f = []
        for p in skypoints:

            assert isinstance(p, Skypoint), "Requires Skypoint objects"
            assert p.coordsystem.is_valid(), "Unsupported coordinate system"
            
            # XLALComputeDetAMResponse() requires equatorial coordinates.
            # We transform the points with Earth-fixed coordsystem ('geographic')
            # to the fiducial equatorial coordinates system with GMST = 0 hr.

            if p.coordsystem.name == 'geographic':
                assert time is None, \
                    'time cannot be forced for skypoints in the geographic coordinate system'
                p = p.transformed_to(FIDUCIAL_EQUATORIAL_COORDSYS_GMST_ZERO)
                gmst_rad = lal.GreenwichMeanSiderealTime(p.coordsystem.ref_time)
            else:
                gmst_rad = lal.GreenwichMeanSiderealTime(time if time is not None else \
                                                     p.coordsystem.ref_time)
            
            # XLALComputeDetAMResponse() takes equatorial coordinates (RA and dec)
            # and gmst to define a sky location
            f.append(lal.ComputeDetAMResponse(self.descriptor.response, \
                                      *p.coords(fmt='lonlat',unit='radians'), \
                                      psi, gmst_rad))

        f = numpy.squeeze(numpy.array(f))
            
        if f.ndim == 1:
            return tuple(f) # fplus, fcross
        else:
            return f[:,0], f[:,1] # fplus, fcross

    def time_delay_from_earth_center(self, skypoints, time=None):
        """ Returns the time delay from the earth center
            skypoints: Skypoint object or list of Skypoint objects (in the same coordinate system)

            This function uses LALTimeDelayFromEarthCenter() that takes equatorial 
            coordinates (RA and dec) and a reference GPS time to define the sky location.

            If the skypoints are given in the equatorial coordinate system, 
            the time delays are computed at the provided time if not None, otherwise
            it is computed at the reference time of the coordinate system of 
            the skypoints.

            If the skypoints are given in the geographic coordinate system,
            they are mapped to fiducial equatorial coordinate system with 
            GMST = 0 hr (REFDATE_GMST_ZERO).

        """

        if not isinstance(skypoints, list):
            skypoints = [skypoints]

        delays = []
        for p in skypoints:

            assert p.coordsystem.is_valid(), "Unsupported coordinate system"
            
            # TimeDelayFromEarthCenter() requires equatorial coordinates.
            # We transform the points with Earth-fixed coordsystem ('geographic')
            # to the fiducial equatorial coordinates system with GMST = 0 hr.

            if p.coordsystem.name == 'geographic':
                p = p.transformed_to(FIDUCIAL_EQUATORIAL_COORDSYS_GMST_ZERO)
                assert time is None, \
                    'time cannot be forced for skypoints in the geographic coordinate system'
                time = p.coordsystem.ref_time
            else:
                time = time if time is not None else p.coordsystem.ref_time

            # TimeDelayFromEarthCenter() takes equatorial coordinates (RA and dec)
            # and a GPS reference time to define a sky location

            delays.append(lal.TimeDelayFromEarthCenter(self.location, \
                                    *p.coords(fmt='lonlat',unit='radians'), time))

        return numpy.squeeze(numpy.array(delays))
        
    def project_strain(self, hplus, hcross, skypoint, psi):
        """ Project hplus and hcross onto the detector frame 
        assuming a given sky location and polarization of the source.
        Apply consistent time delay due to wave propagation.

        hplus, hcross: plus and cross polarisation (REAL8TimeSeries)
        skypoint: Skypoint object
        psi: polarization angle in radians

        This function uses SimDetectorStrainREAL8TimeSeries() that takes equatorial 
        coordinates (RA and dec) and the epoch signal epoch to define the sky location
        and corresponding mapping to an Earth-fixed coordinate system.

        If the skypoint is given in the geographic coordinate system,
        they are mapped to the equatorial coordinate system with
        reference time set to the epoch of hplus and hcross.

        """
        assert hplus.data.length == hcross.data.length, \
                          'The polarization time series must have same size'
        assert hplus.deltaT == hcross.deltaT, \
                          'The polarization time series must have the same sampling step'
        assert hplus.epoch == hcross.epoch, \
                          'The polarization time series must have the same epoch'

        assert skypoint.coordsystem.is_valid(), "Unsupported coordinate system"

        # If the skypoint is in the geographic coordinate system we transform
        # to the equatorial frame at the signal epoch
        if skypoint.coordsystem.name == 'geographic':
            skypoint = skypoint.transformed_to(Coordsystem('equatorial', hplus.epoch))
            
        #print(skypoint.coords(fmt='lonlat',unit='radians'))
            
        return TimeSeries.from_lal(SimDetectorStrainREAL8TimeSeries( \
                                        hplus, hcross, \
                                        *skypoint.coords(fmt='lonlat',unit='radians'), \
                                        psi, self.descriptor))

# def time_delay_rings(detectors, delays):
#     """
#     Return the locii of points with constant delays between each pairs
#     of detectors

#     detectors: tuple of detectors
#     delays: delays from 
#     """
    

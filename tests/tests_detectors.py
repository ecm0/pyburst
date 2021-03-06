from unittest import TestCase

import math
import random
import numpy
from numpy.random import uniform

import lal
from gwpy.timeseries import TimeSeries

import pyburst as pb
import pyburst.detectors, pyburst.skymaps

from pyburst.utils import delayseq

import matplotlib.pyplot as plt

TIME = lal.LIGOTimeGPS(630720013) # Jan 1 2000, 00:00 UTC
COORD_SYS_EQUATORIAL = pb.skymaps.Coordsystem('equatorial', TIME)
COORD_SYS_GEOGRAPHIC = pb.skymaps.Coordsystem('geographic')

SAMPLING_RATE = 4096.0 # Hz
T = numpy.arange(int(SAMPLING_RATE))/SAMPLING_RATE

F0 = 200 # Hz
# phi0 = 0
phi0 = math.radians(uniform(0,360))
COS_1_SEC = numpy.cos(2*math.pi*F0*T + phi0)
SIN_1_SEC = numpy.sin(2*math.pi*F0*T + phi0)
ZEROS_1_SEC = numpy.zeros(int(SAMPLING_RATE))
ZEROS_5_SEC = numpy.zeros(int(5 * SAMPLING_RATE))

DETECTORS = ['H1', 'L1', 'V1']

class TestDetector(TestCase):

    def test_fiducial_equatorial_coordsys(self):
        """ Check that longitude and latitude in the Geographic coord system are equal to 
        right ascension and declination in the fiducial Equatorial coord system"""
        
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt_geo = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_GEOGRAPHIC)
        pt_fiducial = pt_geo.transformed_to(pb.detectors.FIDUCIAL_EQUATORIAL_COORDSYS_GMST_ZERO)

        print(pt_geo)
        print(pt_fiducial)
        
        self.assertAlmostEqual(pt_geo.lon, pt_fiducial.lon)
        self.assertAlmostEqual(pt_fiducial.lat, pt_fiducial.lat)

    def test_antenna_pattern(self):
        """ Check consistency of antenna pattern computed using two coordinate systems
        """
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        pt_geo = pt_eq.transformed_to(COORD_SYS_GEOGRAPHIC)
        d = pb.detectors.Detector(random.choice(DETECTORS))

        pat_eq = d.antenna_pattern(pt_eq)
        pat_geo = d.antenna_pattern(pt_geo)

        self.assertAlmostEqual(pat_eq[0], pat_geo[0]) # fplus
        self.assertAlmostEqual(pat_eq[1], pat_geo[1]) # fcross

        # Force evaluation time to TIME (reference time of
        # the equatorial coord system used here).
        pat_eq = d.antenna_pattern(pt_eq, time=TIME)

        self.assertAlmostEqual(pat_eq[0], pat_geo[0]) # fplus
        self.assertAlmostEqual(pat_eq[1], pat_geo[1]) # fcross

    def test_antenna_pattern_forcing_time_when_geo(self):
        """ Check failure when forcing time with skypoints in the geographic
            coordinate system
        """
        pt = pb.skymaps.Skypoint(0, 0, COORD_SYS_GEOGRAPHIC)
        d = pb.detectors.Detector(random.choice(DETECTORS))

        try:
            pat_geo = d.antenna_pattern(pt, TIME)
        except AssertionError as ae:
            self.assertTrue(ae)
            
    def test_delay(self):
        """ Check consistency of delay computed using two coordinate systems
        """
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        pt_geo = pt_eq.transformed_to(COORD_SYS_GEOGRAPHIC)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        dt_eq = d.time_delay_from_earth_center(pt_eq)
        dt_geo = d.time_delay_from_earth_center(pt_geo)

        self.assertAlmostEqual(dt_eq, dt_geo, places=5)

    def test_delay_forcing_time_when_geo(self):
        """ Check failure when forcing time with skypoints in the geographic
            coordinate system
        """
        pt = pb.skymaps.Skypoint(0, 0, COORD_SYS_GEOGRAPHIC)
        d = pb.detectors.Detector(random.choice(DETECTORS))

        try:
            dt_geo = d.time_delay_from_earth_center(pt, TIME)
        except AssertionError as ae:
            self.assertTrue(ae)
        
    def test_delay_antipodal_point(self):
        """ Check that original and antipodal points have opposite delays
        """

        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        dt_eq = d.time_delay_from_earth_center(pt)
        dt_antipodal = d.time_delay_from_earth_center(pt.antipodal())

        self.assertAlmostEqual(dt_eq, -dt_antipodal, places=5)

    def test_delay_mirror_point(self):
        """ Check that the original and mirror points have same delays
        """

        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        network = [pb.detectors.Detector(d) for d in DETECTORS]
        
        dt_earth_center_orig = numpy.array([d.time_delay_from_earth_center(pt) for d in network])
        dt_earth_center_mirror = numpy.array([d.time_delay_from_earth_center(pt.mirror(network)) \
                                              for d in network])
        
        dt_relative_orig = dt_earth_center_orig[1:] - dt_earth_center_orig[0]
        dt_relative_mirror = dt_earth_center_mirror[1:] - dt_earth_center_mirror[0]

        # print(dt_relative_orig, dt_relative_mirror)

        self.assertTrue(numpy.allclose(dt_relative_orig, dt_relative_mirror))

    def test_delay_ring_isodelays_point(self):
        """ Check that points on the isodelay ring have same delays
        """
        
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        network = [pb.detectors.Detector(d) for d in DETECTORS[:2]]
        
        dt_earth_center_orig = numpy.array([d.time_delay_from_earth_center(pt) for d in network])
        ring = pt.ring_isodelays(network)

        dt_earth_center_ring = []
        for d in network:
            dt_earth_center_ring.append(numpy.array([d.time_delay_from_earth_center(p) \
                                                for p in ring]))
        dt_earth_center_ring = numpy.array(dt_earth_center_ring)
            
        dt_relative_orig = dt_earth_center_orig[1:] - dt_earth_center_orig[0]
        dt_relative_ring = dt_earth_center_ring[1:] - dt_earth_center_ring[0]

        print(dt_relative_orig, dt_relative_ring)

        self.assertTrue(numpy.allclose(dt_relative_orig, dt_relative_ring))
        
    def test_delay_project_strain(self):
        """ Check consistency of project_strain() against time_delay_earth_center()
        """

        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        delay = d.time_delay_from_earth_center(pt_eq, TIME)
    
        hplus = TimeSeries(SIN_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hcross = TimeSeries(ZEROS_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hplus.epoch = lal.LIGOTimeGPS(TIME)
        hcross.epoch = lal.LIGOTimeGPS(TIME)

        # Project wave onto detector
        response = d.project_strain(hplus, hcross, pt_eq, 0)
                
        # Generate support timeseries
        data = TimeSeries(ZEROS_5_SEC, \
                          sample_rate=SAMPLING_RATE, \
                          t0=TIME-2, unit=response._unit)

        # Inject signal into timeseries
        h = data.inject(response)
        
        # Find end of the detector response
        ix, = numpy.where(numpy.abs(h) > numpy.max(h)/10)
        time_end = h.t0.value + ix[-1]/SAMPLING_RATE
        estimated_delay = float(time_end - (TIME+1))
        
        print("Exact delay = {} ; Estimated delay = {}".format(delay, estimated_delay))
        
        # Estimate delay from timeseries
        self.assertAlmostEqual(delay, estimated_delay, places=3)

    def test_fplus_project_strain(self):
        """ Check consistency of project_strain() against antenna_pattern()
        """

        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        psi = math.radians(uniform(0,180))
        
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        antenna_pat = d.antenna_pattern(pt_eq, time=TIME, psi=psi)
        
        hplus = TimeSeries(SIN_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hcross = TimeSeries(ZEROS_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hplus.epoch = lal.LIGOTimeGPS(TIME)
        hcross.epoch = lal.LIGOTimeGPS(TIME)
            
        # Project wave onto detector
        response = d.project_strain(hplus, hcross, pt_eq, psi)
                
        # Generate support timeseries
        data = TimeSeries(ZEROS_5_SEC, \
                          sample_rate=SAMPLING_RATE, \
                          t0=TIME-2, unit=response._unit)

        # Inject signal into timeseries
        h = data.inject(response)

        if antenna_pat[0] > 0:
            estimated_pat = h.max().to_value()
        else:
            estimated_pat = h.min().to_value()

        print("Exact antenna pattern = {} ; Estimated amplitude = {}".format(antenna_pat[0], estimated_pat))
            
        # Estimate delay from timeseries
        self.assertAlmostEqual(antenna_pat[0], estimated_pat, places=2)    

    def test_fcross_project_strain(self):
        """ Check consistency of project_strain() against antenna_pattern()
        """

        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        psi = math.radians(uniform(0,180))
        
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        antenna_pat = d.antenna_pattern(pt_eq, time=TIME, psi=psi)
        
        hplus = TimeSeries(ZEROS_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hcross = TimeSeries(SIN_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hplus.epoch = lal.LIGOTimeGPS(TIME)
        hcross.epoch = lal.LIGOTimeGPS(TIME)
            
        # Project wave onto detector
        response = d.project_strain(hplus, hcross, pt_eq, psi)
                
        # Generate support timeseries
        data = TimeSeries(ZEROS_5_SEC, \
                          sample_rate=SAMPLING_RATE, \
                          t0=TIME-2, unit=response._unit)

        # Inject signal into timeseries
        h = data.inject(response)

        if antenna_pat[1] > 0:
            estimated_pat = h.max().to_value()
        else:
            estimated_pat = h.min().to_value()

        print("Exact antenna pattern = {} ; Estimated pattern from amplitude = {}".format(antenna_pat[1], estimated_pat))
            
        self.assertAlmostEqual(antenna_pat[1], estimated_pat, places=2)

        # Test sky points in grid have similar antennna patterns and delays

# close_pixels = healpy.pixelfunc.get_interp_weights(sky.nside,*pt_geo.coords(fmt='colatlon'),sky.order)
# for ix in close_pixels[0]:
#    print('ix={}: patterns: {}, delays: {}'.format(ix,antenna_patterns[ix],delays[ix]))

# lon, lat = sky.grid.healpix_to_lonlat(range(sky.grid.npix))

# fig, axes = plt.subplots()
# for lo,la in zip(lon,lat):
#    plt.plot(lo, la,'g.')
# plt.plot(pt_geo.lon, pt_geo.lat,'rx')
# for ix in close_pixels[0]:
#     print(ix)
#     plt.plot(lon[ix], lat[ix],'b.')

    def test_coordframe_project_strain(self):
        """ Check consistency of project_strain() with skypoints in different cooordinate frames
        """

        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        pt_geo = pt_eq.transformed_to(COORD_SYS_GEOGRAPHIC)
        
        d = pb.detectors.Detector(random.choice(DETECTORS))
    
        hplus = TimeSeries(SIN_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hcross = TimeSeries(ZEROS_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        hplus.epoch = lal.LIGOTimeGPS(TIME)
        hcross.epoch = lal.LIGOTimeGPS(TIME)

        # Project wave onto detector
        response_eq = d.project_strain(hplus, hcross, pt_eq, 0)
        response_geo = d.project_strain(hplus, hcross, pt_geo, 0)

        err = numpy.abs(response_eq-response_geo)

        # self.assertEqual(response_eq, response_geo)
        self.assertTrue(numpy.allclose(numpy.abs(err), 0))
        
        # def test_global_project_strain(self):
        #     """ Global consistency check for project_strain() against ad-hoc response computation
        #         using delay_seq() and antenna_pattern()
        #     """
        
        #     coords = numpy.array([uniform(0,360), uniform(-90,90)])
        #     psi = math.radians(uniform(0,180))
        
        #     pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        #     d = pb.detectors.Detector(random.choice(DETECTORS))
        #     antenna_pat = d.antenna_pattern(pt_eq, time=TIME, psi=psi)
        #     delay = d.time_delay_from_earth_center(pt_eq, TIME)
        
        #     hplus = TimeSeries(COS_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        #     hcross = TimeSeries(SIN_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
        #     hplus.epoch = lal.LIGOTimeGPS(TIME)
        #     hcross.epoch = lal.LIGOTimeGPS(TIME)
            
        #     # Project wave onto detector
        #     response = d.project_strain(hplus, hcross, pt_eq, psi)
    
        #     # Compute ad-hoc response using delayseq()
        #     resp_adhoc = antenna_pat[0] * delayseq(COS_1_SEC, delay * SAMPLING_RATE) + \
        #                  antenna_pat[1] * delayseq(SIN_1_SEC, delay * SAMPLING_RATE)

        # Compare the two responses
        # self.assertTrue(numpy.allclose(numpy.abs(response-resp_adhoc), 0))

        # ax = plt.subplot(2, 2, 1)
        # plt.plot(response.data, label='orig')
        # ax = plt.subplot(2, 2, 2)
        # ax.plot(resp_adhoc, label='ad-hoc')
        # ax.legend()

        # plt.show()

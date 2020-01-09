from unittest import TestCase

import math
import random
import numpy
from numpy.random import uniform

import lal
from gwpy.timeseries import TimeSeries

import pyburst as pb
import pyburst.detectors, pyburst.skymaps

import matplotlib.pyplot as plt

TIME = lal.LIGOTimeGPS(630720013) # Jan 1 2000, 00:00 UTC
COORD_SYS_EQUATORIAL = pb.skymaps.Coordsystem('equatorial', TIME)
COORD_SYS_GEOGRAPHIC = pb.skymaps.Coordsystem('geographic', TIME)

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

    def test_antenna_pattern(self):
        """ Check consistency of antenna pattern computed using two coordinate systems
        """
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        pt_geo = pt_eq.transformed_to(COORD_SYS_GEOGRAPHIC)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        pat_eq = d.antenna_pattern(pt_eq, ref_time=TIME)
        pat_geo = d.antenna_pattern(pt_geo, ref_time=None)

        self.assertAlmostEqual(pat_eq[0], pat_geo[0]) # fplus
        self.assertAlmostEqual(pat_eq[1], pat_geo[1]) # fcross

    def test_delay(self):
        """ Check consistency of delay computed using two coordinate systems
        """
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt_eq = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        pt_geo = pt_eq.transformed_to(COORD_SYS_GEOGRAPHIC)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        dt_eq = d.time_delay_from_earth_center(pt_eq, ref_time=TIME)
        dt_geo = d.time_delay_from_earth_center(pt_geo)

        self.assertAlmostEqual(dt_eq, dt_geo, places=5)

    def test_delay_mirror_point(self):
        """ Check that mirror point has opposite delays
        """
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt = pb.skymaps.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)
        d = pb.detectors.Detector(random.choice(DETECTORS))
        dt_eq = d.time_delay_from_earth_center(pt, ref_time=TIME)
        dt_mirror = d.time_delay_from_earth_center(pt.mirror(), ref_time=TIME)

        self.assertAlmostEqual(dt_eq, -dt_mirror, places=5)
        
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
        antenna_pat = d.antenna_pattern(pt_eq, ref_time=TIME, psi=psi)
        
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
        antenna_pat = d.antenna_pattern(pt_eq, ref_time=TIME, psi=psi)
        
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

        print("Exact antenna pattern = {} ; Estimated amplitude = {}".format(antenna_pat[0], estimated_pat))
            
        # Estimate delay from timeseries
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
    #     antenna_pat = d.antenna_pattern(pt_eq, ref_time=TIME, psi=psi)
    #     delay = d.time_delay_from_earth_center(pt_eq, TIME)
        
    #     hplus = TimeSeries(COS_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
    #     hcross = TimeSeries(SIN_1_SEC, sample_rate=SAMPLING_RATE).to_lal()
    #     hplus.epoch = lal.LIGOTimeGPS(TIME)
    #     hcross.epoch = lal.LIGOTimeGPS(TIME)
            
    #     # Project wave onto detector
    #     response = d.project_strain(hplus, hcross, pt_eq, psi)
                
    #     # Generate support timeseries
    #     data = TimeSeries(ZEROS_5_SEC, \
    #                       sample_rate=SAMPLING_RATE, \
    #                       t0=TIME-2, unit=response._unit)

    #     # Inject signal into timeseries
    #     h = data.inject(response)

    #     # Compute ad-hoc response using delayseq()
    #     h_adhoc = antenna_pat[0] * delayseq(hplus, delay * SAMPLING_RATE) + \
    #               antenna_pat[1] * delayseq(hcross, delay * SAMPLING_RATE) + \

    #     # Compare the two responses
    #     self.assertTrue(numpy.allclose(numpy.abs(h-h_adhoc), 0))


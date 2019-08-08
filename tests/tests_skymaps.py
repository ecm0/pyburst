from unittest import TestCase

import math
import numpy
from numpy.random import uniform
import healpy

import pyburst.skymaps as pb

NSIDE = 32
TIME = LIGOTimeGPS(630720013) # Jan 1 2000, 00:00 UTC
COORD_SYS = 'equatorial'

class TestCoordsystem(TestCase):

    def test_initcoordsystem(self):
        try:
            pb.Coordsystem('geographic')
        except Exception:
            self.fail('Coordsystem instantiation failed')
        try:
            pb.Coordsystem('equatorial')
        except Exception:
            self.fail('Coordsystem instantiation failed')
        try:
            pb.Coordsystem('test')
        except Exception as ex:
            self.assertTrue('Unsupported' in ex.args[0])
        
        # self.assertRaises(AssertionError, pb.Coordsystem, 'test')            

class TestSkypoint(TestCase):

    def test_initskypoint(self):
        try:
            p = pb.skymaps.Skypoint(0, 0, 'equatorial', '')
        except Exception:
            self.fail('Skypoint instantiation failed')
        try:
            p = pb.skymaps.Skypoint(0, 0, 'geographic', '')
        except Exception:
            self.fail('Skypoint instantiation failed')


    def test_conversion(self):
        
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt_orig = pb.skymaps.Skypoint(*numpy.radians(coords), 'equatorial', '')
        pt_geo = pt_orig.transformed_to('geographic', TIME)
        pt_eq = pt_geo.transformed_to('equatorial', TIME)

        self.assertSequenceEqual((pt_orig.lon, pt_orig.lat), (pt_eq.lon, pt_eq.lat))
        
class TestSkymap(TestCase):

    def test_value(self):
        ''' Test Skymap.value()'''

        # Generate a random skypoint
        p = pb.Skypoint(numpy.random.uniform(high=2.*math.pi), \
             numpy.random.uniform(low=-math.pi/2,high=math.pi/2), COORD_SYS)

        # Generate skymap and set all pixel values to zero
        zeros = numpy.zeros(healpy.nside2npix(NSIDE))
        sky = pb.Skymap(NSIDE, COORD_SYS, zeros, order='nested')

        # Set value of selected pixel to 1.0
        idx = healpy.ang2pix(NSIDE,*p.coords(),nest=True)
        sky.data[idx] = 1.0
        
        self.assertEqual(sky.value(p), 1.0)


        pt_eq = pb.skymaps.Skypoint(numpy.radians(ra), numpy.radians(dec), \
                     'equatorial', 'injection')
print(pt_eq)
pt_geo = pt_eq.transform_to('geographic', time)
print(pt_geo)

# Unit test to check consistency of antenna pattern and delays computed using equat and geom coords
ra = 90.0
dec = 10.0
time = float(1187008887)
# t_ref = lal.LIGOTimeGPS(630696086, 238589290) # Dec 31 1999, 17:21:13 238589
pt_eq = pb.skymaps.Skypoint(numpy.radians(ra), numpy.radians(dec), 'equatorial', 'point')
pt_geo = pt_eq.transform_to('geographic', time)
print('\n'.join(map(str,[pt_eq, pt_geo])))
print(network[0].antenna_pattern(*pt_eq.coords(fmt='lonlat', unit='radians'), 0, ref_time=time))
print(network[0].antenna_pattern(*pt_geo.coords(fmt='lonlat', unit='radians'), 0, ref_time=None))
target = network[0].time_delay_from_earth_center(*pt_eq.coords(fmt='lonlat', unit='radians'), ref_time=time)
print(target)
res1 = network[0].time_delay_from_earth_center(*pt_geo.coords(fmt='lonlat', unit='radians'), ref_time=t_ref)
res2 = network[0].time_delay_from_earth_center(*pt_geo.coords(fmt='lonlat', unit='radians'), ref_time=None)
print(res1, res2)

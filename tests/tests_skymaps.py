from unittest import TestCase

import math
import numpy
from numpy.random import uniform
import healpy

import lal
import pyburst.skymaps as pb

NSIDE = 32
TIME = lal.LIGOTimeGPS(630720013) # Jan 1 2000, 00:00 UTC
COORD_SYS_EQUATORIAL = pb.Coordsystem('equatorial', TIME)
COORD_SYS_GEOGRAPHIC = pb.Coordsystem('geographic', TIME)

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
            p = pb.Skypoint(0, 0, COORD_SYS_EQUATORIAL)
        except Exception:
            self.fail('Skypoint instantiation failed')
        try:
            p = pb.Skypoint(0, 0, COORD_SYS_GEOGRAPHIC)
        except Exception:
            self.fail('Skypoint instantiation failed')


    def test_reversibity(self):
        """ Check reversibility of coordinate conversion
        """
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        pt_orig = pb.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL, '')
        pt_geo = pt_orig.transformed_to(COORD_SYS_GEOGRAPHIC)
        pt_eq = pt_geo.transformed_to(COORD_SYS_EQUATORIAL)

        self.assertAlmostEqual(pt_orig.lon, pt_eq.lon, places=7)
        self.assertAlmostEqual(pt_orig.lat, pt_eq.lat, places=7)
        
class TestSkymap(TestCase):

    def test_value(self):
        ''' Test Skymap.value()'''

        # Generate a random skypoint
        coords = numpy.array([uniform(0,360), uniform(-90,90)])
        p = pb.Skypoint(*numpy.radians(coords), COORD_SYS_EQUATORIAL)

        # Generate skymap and set all pixel values to zero
        zeros = numpy.zeros(healpy.nside2npix(NSIDE))
        sky = pb.Skymap(NSIDE, COORD_SYS_EQUATORIAL, order='nested', array=zeros)

        # Set value of selected pixel to 1.0
        idx = healpy.ang2pix(NSIDE,*p.coords(),nest=True)
        sky.data[idx] = 1.0
        
        self.assertEqual(sky.value(p), 1.0)


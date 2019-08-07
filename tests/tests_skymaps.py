from unittest import TestCase

import math
import numpy
import healpy

import pyburst.skymaps as pb

NSIDE = 32
COORD_SYS = 'equatorial'

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

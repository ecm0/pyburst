
import math
import copy
import numpy
import logging

import lal
import healpy
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS
import matplotlib.pyplot as plt

COORD_SYSTEMS = {'geographic': lal.COORDINATESYSTEM_GEOGRAPHIC, \
                 'equatorial': lal.COORDINATESYSTEM_EQUATORIAL}

COORD_TRANSFORMS = { \
    ('geographic','equatorial'): lal.GeographicToEquatorial, \
    ('equatorial','geographic'): lal.EquatorialToGeographic
}

PRETTY_PRINT_POINT_STR = "{} ({}): lon= {:>+1.8f} rad ({:7.2f} deg) lat= {:>+1.8f} rad ({:7.2f} deg)"

class Coordsystem(object):
        """
        A Coordsystem object characterizes a coordinate system of the celestial sphere.

        The geocentric Earth-fixed coordinate system is the primary reference and doesn't
        not require a reference time. Sky-fixed coordinate systems are considered to move
        in time, and are characterized by a reference time. This reference will be used
        to transform between coordinate systems.
        """
    
        def __init__(self, name, ref_time=None):
            """
            name -- name of the coordinate system
            ref_time: reference time for sky-fixed coordinate systems
            ref_time is None for the Earth-fixed (geocentric) coordinate system.
            """

            self.name = name
            assert self.is_valid(), "Unsupported coordinate system"
            
            if name=='geographic':
                assert ref_time is None, \
                    'ref_time must be None for Earth-fixed coordinate system'
            else:
                assert ref_time is not None, \
                    'ref_time must be provided for sky-fixed coordinate systems'
            
            self.ref_time = ref_time
            
        def __str__(self):
            return self.name
        
        def is_valid(self):
            return self.name in COORD_SYSTEMS.keys()

        def is_same(self, coordsystem):
            return self.name==coordsystem.name and self.ref_time==coordsystem.ref_time

        def has_same_type_as(self, coordsystem):
            return self.name==coordsystem.name
        
        def to_lal(self):
            return COORD_SYSTEMS[self.name]
        
        def transforms_to(self, target):
            """
            Returns the coordinate transformation function from self to target
            """

            assert (self.name,target.name) in COORD_TRANSFORMS.keys(), "Unsupported coordinate system transformation"
            return COORD_TRANSFORMS[(self.name,target.name)]

class Skypoint(object):
    """
    A Skypoint object describes a direction in the sky in a given coordinate system.
    """
    
    def __init__(self, lon, lat, coordsystem, label=''):
        """
        Instantiate a SkyPoint from spherical coordinates

        lon -- longitude [geographic] or right ascension [equatorial] (in radians)
        lat -- latitude [geographic] or declination [equatorial] (in radians)
        coordsystem -- Coordsystem object that describes the coordinate system
        label -- optional label
        """
        
        assert coordsystem.is_valid(),  "Unsupported coordinate system"
        
        self.lon = lon
        self.lat = lat
        self.coordsystem = coordsystem
        self.label = label
        
    @classmethod
    def from_cart(cls, xyz, coordsystem, label=''):
        """
        Instantiate a SkyPoint from cartesian coordinates
        The input coordinates must be on the unit sphere
        
        xyz -- tuple or Numpy array with the X, Y and Z coordinates
        coordsystem -- Coordsystem object that describes the coordinate system
        label -- optional qualifying label        
        """

        assert coordsystem.is_valid(),  "Unsupported coordinate system"
        assert math.isclose(numpy.linalg.norm(xyz),1), "Input point should be on the unit sphere"

        return cls(numpy.arctan2(xyz[1], xyz[0]), \
                   numpy.arctan2(xyz[2], numpy.sqrt(xyz[0]**2 + xyz[1]**2)), \
                   coordsystem, label)
        
    def __str__(self):
            return PRETTY_PRINT_POINT_STR.format(self.label, self.coordsystem, \
                                            self.lon, numpy.degrees(self.lon),\
                                            self.lat, numpy.degrees(self.lat))
        
    def coords(self, fmt='colatlon', unit='radians'):
        """
        Return the skypoint coordinates in different format and units.
        When the format is 'colatlon' (default), the co-latitude and 
        longitude is returned. 
        Available formats are 'colatlon', 'latlon', 'lonlat' and 'cart'.
        """
        
        if fmt=='latlon':
            coords = (self.lat, self.lon)
        elif fmt=='colatlon': 
            coords = (math.pi/2-self.lat, self.lon)
        elif fmt=='lonlat':
            coords = (self.lon, self.lat)
        elif fmt=='cart' or fmt=='cartesian':
            return (math.cos(self.lat) * math.cos(self.lon), \
                    math.cos(self.lat) * math.sin(self.lon), \
                    math.sin(self.lat))
        else:
            logging.warning('Unknown format')
            return None
        
        if unit=='radians':
            return coords
        elif unit=='degrees':
            return map(math.degrees, coords)
        else:
            logging.warning('Unknown unit')
            return None
            
    def transformed_to(self, coordsystem, label=''):
        """
        Transforms the Skypoint object's coordinates to another coordinate system

        """
    
        if self.coordsystem.has_same_type_as(coordsystem):
            return Skypoint(self.lon, self.lat, coordsystem, self.label)
        
        input = lal.SkyPosition()
        input.longitude = self.lon
        input.latitude = self.lat
        input.system = self.coordsystem.to_lal()

        output = lal.SkyPosition()

        ref_times = [t for t in [self.coordsystem.ref_time, coordsystem.ref_time] \
                         if t is not None]
        assert len(ref_times)==1, "There should be only one reference time"
            
        target = lal.ConvertSkyParams()
        target.system = coordsystem.to_lal()
        target.gpsTime = ref_times[0]

        lal.ConvertSkyCoordinates(output, input, target)
        # self.coordsystem.transforms_to(coordsystem)(output,input,coordsystem.ref_time)
        
        label = " ".join((self.label, label)) if label else self.label
        return Skypoint(output.longitude, output.latitude, coordsystem, label)

    def antipodal(self):
        return Skypoint((self.lon + math.pi) % (2 * math.pi), -self.lat, \
                        self.coordsystem, "Antipodal point to " + self.label)

    def mirror(self, detector_triplet):
        """
        Compute so-called mirror point with same time-of-flight delays in a
        network with three detectors
        """
        assert len(detector_triplet)==3, "Requires a tuple with 3 detectors"
        
        ref = detector_triplet[0]
        baselines = [d.location-ref.location for d in detector_triplet[1:]]
        normal_detector_plane = numpy.cross(baselines[0], baselines[1])
        normal_detector_plane /= numpy.linalg.norm(normal_detector_plane)

        source = self.transformed_to(Coordsystem('geographic')).coords('cart')
        
        mirror = source - 2 * (normal_detector_plane @ source) * normal_detector_plane
        mirror /= numpy.linalg.norm(mirror)

        return Skypoint.from_cart(mirror, Coordsystem('geographic')) \
                       .transformed_to(self.coordsystem, "Mirror of " + self.label)
    
    def display(self, marker, color):
        healpy.projplot(*self.coords(), marker+color, \
                        markersize=6, \
                        label=self.label)
        
#        healpy.projplot(*self.coords_deg(), marker+color, \
#                        markersize=6, \
#                        label=self.label, \
#                        lonlat=True)


class Skymap(object):
    """ 
    A skymap object is an HEALPix map equipped with a custom coordinate system -- from LAL.
    """
    
    def __init__(self, nside, coordsystem, order='nested', array=None):
        """
        grid: HEALPix map object
        nside: HEALPix nside parameter (int, power of 2)
        order: pixel ordering scheme of the HEALPix map
        coordsystem: Coordsystem object that describes the coordinate system  
        Note: 
        The HEALPix map used for the grid is initialized with a dummy ICRS 
        coordinate frame, that is not used in practise as healpy nor astropy 
        don't support ECEF/Geographic coordinate systems. 
        Instead the grid coordinates are defined through a coordinate system
        obtained from the LAL library.
        """

        if array is not None:
            assert len(array) == healpy.nside2npix(nside), "Data array has incorrect size"
        
        self.grid = HEALPix(nside=nside, order=order, frame=ICRS()) 
        self.nside = nside
        self.order = order
        self.coordsystem = coordsystem
        self.data = numpy.empty(healpy.nside2npix(nside)) if array is None else array

    @classmethod
    def like(cls, skymap, coordsystem=None, data=None):
        """
        Instantiate a Skymap object with the same feature as the input
        skymap unless otherwise requested.
        
        skymap: reference Skymap object
        coordsystem: Coordsystem object that defines the coordinate system 
                     of the new Skymap object 
        data: data of the new Skymap object 
        """

        if coordsystem is not None:
            assert coordsystem.is_valid(),  "Unsupported coordinate system"
            
        if data is not None:
            assert len(data) == skymap.grid.npix, 'Input data has wrong dimensions'
            
        return cls(skymap.nside, \
                   coordsystem if coordsystem is not None else skymap.coordsystem, \
                   skymap.order, \
                   data if data is not None else skymap.data)
                   
    def is_nested(self):
        return self.order == 'nested'

    def grid_points(self):
        return [Skypoint(math.radians(p.ra.value), math.radians(p.dec.value), self.coordsystem) \
                for p in self.grid.healpix_to_skycoord(range(self.grid.npix))]

    def feed(self, data):
        """
        Update the skymap data with the given array
        """

        assert len(data) == self.grid.npix, 'Input data has wrong dimensions'
        out.data = data

    def value(self, skypoint):
        """
        Returns the skymap value at a skypoint. The skymap and skypoint
        must have the same coordinate system.
        """
        
        assert self.coordsystem.is_same(skypoint.coordsystem), \
            "The skymap and skypoint must have the same coordinate system"
        
        # Get the pixel index that corresponds to the skypoint coordinates  
        idx = healpy.ang2pix(self.nside, \
                                *skypoint.coords(), \
                                nest=self.is_nested())
        return self.data[idx]
    
    def transformed_to(self, coordsystem):
        """
        Transforms the skymap to another coordinate system
        """

        if self.coordsystem.has_same_type_as(coordsystem):
            return Skymap.like(self, coordsystem=coordsystem)
        
        # Get co-latitude and longitude coordinates of the skygrid in the target coord system 
        colat, lon = healpy.pix2ang(self.nside, \
                                    numpy.arange(healpy.nside2npix(self.nside)), \
                                    nest=self.is_nested())
        
        # Map target to original coordinates
        points = [Skypoint(l, math.pi/2-c, coordsystem).transformed_to(self.coordsystem) \
                                  for l, c in zip(lon, colat)]
        
        # Transform the list of coordinate tuples [(lon, lat), ...] into two lists [lons, lats]
        coords_rot = [p.coords() for p in points]
        (colat_rot, lon_rot) = tuple(map(list,zip(*coords_rot)))
 
        # Compute skymap in target coord system: interpolate the skymap at points that map to target coords
        data_rotated = healpy.get_interp_val(self.data, numpy.array(colat_rot), numpy.array(lon_rot), \
                                             nest=self.is_nested())
        
        # Create target skymap and update its data 
        return Skymap.like(self, coordsystem=coordsystem, data=data_rotated)
    
    def argmin(self, label = 'skymap minimum'):
        """
        Returns the location of the skymap minimum
        """
        
        coords = healpy.pix2ang(self.nside,numpy.argmin(self.data), nest=self.is_nested())
        return Skypoint(coords[1], math.pi/2-coords[0], self.coordsystem, \
                             label + " (val={0:.2f})".format(numpy.min(self.data)))
    
    def display(self, title, cmap_name='gray_r'):
        """
        Display the skymap
        """
        
        cmap = plt.get_cmap(cmap_name)
        cmap.set_under('w')
        healpy.mollview(self.data, \
                    nest=self.is_nested(), cmap=cmap, \
                    title=title + " ({})".format(self.coordsystem))

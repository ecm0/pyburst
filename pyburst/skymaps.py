
import math
import copy
import numpy
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
        A Coordsystem object characterizes a coordinate system of the celestial sphere
        """
    
        def __init__(self, name, ref_time=None):
            """
            name -- name of the coordinate systems
            ref_time: reference time when using the equatorial coordinate system
            """
            
            self.name = name
            self.ref_time = ref_time
            self.is_valid()
            
        def __str__(self):
            return self.name
        
        def is_valid(self):
            assert self.name in COORD_SYSTEMS.keys(), "Unsupported coordinate system"
        
        def to_lal(self):
            return COORD_SYSTEMS[self.name]
        
        def transforms_to(self, target):
            """
            Returns the coordinate transformation function from self to target
            """
            
            assert (self.name,target.name) in COORD_TRANSFORMS.keys(), "Unsupported coord transformation"
            return COORD_TRANSFORMS[(self.name,target.name)]

class Skypoint(object):
    """
    A Skypoint object describes a direction in the sky in a given coordinate system.
    """
    
    def __init__(self, lon, lat, coordsystem, label=''):
        """
        lon -- longitude or right ascension (in radians)
        lat -- latitude or declination (in radians)
        coordsystem -- Coordsystem object that describes the coordinate system
        label -- optional qualifying label
        """
        
        coordsystem.is_valid()
        
        self.lon = lon
        self.lat = lat
        self.coordsystem = coordsystem
        self.label = label
    
    def __str__(self):
            return PRETTY_PRINT_POINT_STR.format(self.label, self.coordsystem, \
                                            self.lon, numpy.degrees(self.lon),\
                                            self.lat, numpy.degrees(self.lat))
        
    def coords(self, fmt='colatlon', unit='radians'):
        """
        Return the skypoint coordinates in different format and units.
        When the format is 'colatlon' (default), a tuple with
        the co-latitude and longitude is returned. 
        """
        
        if fmt=='latlon':
            angles = (self.lat, self.lon)
        elif fmt=='colatlon': 
            angles = (math.pi/2-self.lat, self.lon)
        elif fmt=='lonlat':
            angles = (self.lon, self.lat)
        else:
            logging.warning('Unknown format')
            return None
        
        if unit=='radians':
            return angles
        elif unit=='degrees':
            return map(math.degrees, angles)
        else:
            logging.warning('Unknown unit')
            return None
            
    def transformed_to(self, coordsystem, label=''):
        """
        Transforms to another coordinates system
        """
    
        if self.coordsystem == coordsystem:
            logging.warning('Attempt to transform to same coordinate system')
            return self
                     
        input = lal.SkyPosition()
        input.longitude = self.lon
        input.latitude = self.lat
        input.system = self.coordsystem.to_lal()

        output = lal.SkyPosition()
        
        self.coordsystem.transforms_to(coordsystem)(output,input,coordsystem.ref_time)

        label = " ".join(self.label, label) if label else self.label
        return Skypoint(output.longitude, output.latitude, coordsystem, label)

    def mirror(self):
        return Skypoint((self.lon + math.pi) % (2 * math.pi), -self.lat, \
                        self.coordsystem, "Mirror of " + self.label)
    
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
            
        self.grid = HEALPix(nside=nside, order=order, frame=ICRS()) 
        self.nside = nside
        self.order = order
        self.coordsystem = coordsystem
        self.data = numpy.empty(healpy.nside2npix(nside)) if array is None else array
        
    def is_nested(self):
        return self.order == 'nested'

    def grid_points(self):
        return [Skypoint(math.radians(p.ra.value), math.radians(p.dec.value), self.coordsystem) \
                for p in self.grid.healpix_to_skycoord(range(self.grid.npix))]

    def feed(self, data):
        out = copy.copy(self)
        out.data = data
        return out

    def value(self, skypoint):
        """
        Returns the skymap value at a skypoint. The skymap and skypoint
        must have the same coordinate system.
        """
        
        assert self.coordsystem.name == skypoint.coordsystem.name, "The skymap and skypoint must have the same coordinate system"
        
        # Get the pixel index that corresponds to the skypoint coordinates  
        idx = healpy.ang2pix(self.nside, \
                                *skypoint.coords(), \
                                nest=self.is_nested())
        return self.data[idx]
    
    def transformed_to(self, coordsystem):
        """
        Transforms the skymap to another coordinate system
        """
        
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
        data_rot = healpy.get_interp_val(self.data, numpy.array(colat_rot), numpy.array(lon_rot), \
                                             nest=self.is_nested())
        
        # Create target skymap and set its coordinate system
        out = self.feed(data_rot)
        out.coordsystem = coordsystem
 
        return out
    
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

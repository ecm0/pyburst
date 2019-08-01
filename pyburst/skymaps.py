import numpy
import healpy
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, ICRS


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
    
        def __init__(self, name):
            
            assert name in COORD_SYSTEMS.keys(), "Unsupported coordinate system"
            self.name = name
    
        def __str__(self):
            return self.name
        
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
    
    def __init__(self, lon, lat, system_name, label=''):
        """
        lon -- longitude or right ascension (in radians)
        lat -- latitude or declination (in radians)
        system_name -- name of the coordinate system (str)
        label -- optional qualifying label
        """
        
        self.lon = lon
        self.lat = lat
        self.system = Coordsystem(system_name)
        self.label = label
    
    def __str__(self):
            return PRETTY_PRINT_POINT_STR.format(self.label, self.system, \
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
            
    def transform_to(self, system_name, time):
        """
        Transforms to another coordinates system
        """
    
        system = Coordsystem(system_name)
    
        if self.system == system:
            logging.warning('Attempt to transform to same coordinate system')
            return self
                     
        input = lal.SkyPosition()
        input.longitude = self.lon
        input.latitude = self.lat
        input.system = self.system.to_lal()

        output = lal.SkyPosition()

        self.system.transforms_to(system)(output,input,time)
        
        return Skypoint(output.longitude, output.latitude, system.name, self.label)
    
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
    
    def __init__(self, nside, system_name, array, order='nested'):
        """
        """
            
        # HEALPix map is defined in the ICRS() frame, but the 
        # HEALPix frame is not used in practice.
        # The coordinate system that is used is defined by self.system
        # Note: healpy nor astropy do not support ECEF/Geographic coordinate systems (yet)
        # The skymap is equipped with a custom coordinate system.

        self.grid = HEALPix(nside=nside, order=order, frame=ICRS()) 
        self.data = array
        self.nside = nside
        self.order = order
        self.system = Coordsystem(system_name)
       
    def is_nested(self):
        return self.order == 'nested'
    
    def feed(self, data):
        out = copy.copy(self)
        out.data = data
        return out
    
    def value(self, skypoint):
        """
        Returns the skymap value at a skypoint. The skymap and skypoint
        must have the same coordinate system.
        """
        
        assert self.system.name == skypoint.system.name, "The skymap and skypoint must have the same coordinate system"
        
        # Get the pixel index that corresponds to the skypoint coordinates  
        idx = healpy.ang2pix(self.nside, \
                                *skypoint.coords(), \
                                nest=self.is_nested())
        return self.data[idx]
    
    def transform_to(self, system_name, time):
        """
        Transforms the skymap to another coordinate system
        """
        
        # Get co-latitude and longitude coordinates of the skygrid in the target coord system 
        colat, lon = healpy.pix2ang(self.nside, \
                                    np.arange(healpy.nside2npix(self.nside)), \
                                    nest=self.is_nested())
        
        # Map target to original coordinates
        points = [Skypoint(l, math.pi/2-c, system_name).transform_to(self.system.name, time) \
                                  for l, c in zip(lon, colat)]
        
        # Transform the list of coordinate tuples [(lon, lat), ...] into two lists [lons, lats]
        coords_rot = [p.coords() for p in points]
        (colat_rot, lon_rot) = tuple(map(list,zip(*coords_rot)))
 
        # Compute skymap in target coord system: interpolate the skymap at points that map to target coords
        data_rot = healpy.get_interp_val(self.data, np.array(colat_rot), np.array(lon_rot), \
                                             nest=self.is_nested())
        
        # Create target skymap and set its coordinate system
        out = self.feed(data_rot)
        out.system = Coordsystem(system_name)
 
        return out
    
    def argmin(self, label = 'skymap minimum'):
        """
        Returns the location of the skymap minimum
        """
        
        coords = healpy.pix2ang(self.nside,np.argmin(self.data), nest=self.is_nested())
        return Skypoint(coords[1], math.pi/2-coords[0], self.system.name, \
                             label + " (val={0:.2f})".format(np.min(self.data)))
    
    def display(self, title, cmap_name='gray_r'):
        """
        Display the skymap
        """
        
        cmap = plt.get_cmap(cmap_name)
        cmap.set_under('w')
        healpy.mollview(self.data, \
                    nest=self.is_nested(), cmap=cmap, \
                    title=title + " ({})".format(self.system))

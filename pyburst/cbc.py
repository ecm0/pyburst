import math

import lal
from lal import MSUN_SI as LAL_MSUN_SI
from lal import PC_SI as LAL_PC_SI
from lal import DictInsertREAL8Value, DictInsertINT4Value

import lalsimulation
from lalsimulation import SimInspiralTD, SimInspiralCreateWaveformFlags, \
                            GetApproximantFromString, SimInspiralWaveformParamsInsertTidalLambda1, \
                            SimInspiralWaveformParamsInsertTidalLambda2

ZERO = {'x': 0., 'y': 0., 'z': 0.}

# map order integer to a string that can be parsed by lalsimulation
PN_ORDERS = {
    'default'          : -1,
    'zeroPN'           : 0,
    'onePN'            : 2,
    'onePointFivePN'   : 3,
    'twoPN'            : 4,
    'twoPointFivePN'   : 5,
    'threePN'          : 6,
    'threePointFivePN' : 7,
    'pseudoFourPN'     : 8,
    }

class Binary(object):
    """
    A CompactBinary object characterises a binary formed by two compact objects.
    """

    def __init__(self, mass1=1.4, mass2=1.4, distance=1, \
                 spin1=ZERO, spin2=ZERO, lambda1=0, lambda2=0, 
                 eccentricity=0, meanPerAno=0, 
                 inclination=0, psi=0):
        """
        mass1, mass2 -- masses of the binary components in solar masses
        distance -- distance of the binary in Mpc
        redshift -- redshift of the binary. If zero, cosmology is ignored.
        spin1, spin2 -- spin vectors of binary components
        lambda1,lambda2 -- tidal parameters
        eccentricity -- eccentricity at reference epoch
        meanPerAno -- mean anomaly of periastron
        inclination -- inclination angle with respect to the line of sight in degrees
        psi -- longitude of ascending nodes, degenerate with the polarization angle (Omega or psi)
        """
        self.mass1 = mass1
        self.mass2 = mass2
        self.distance = distance
        self.spin1 = spin1
        self.spin2 = spin2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eccentricity = eccentricity
        self.meanPerAno = meanPerAno 
        self.iota = inclination
        self.longAscNodes = psi
        
class Template(object):
    """
    A Template object characterises the gravitational
    wave (GW) chirp signal associated to the coalescence of two
    inspiralling compact objects.
    """
        
    def __init__(self, approximant, amplitude0, phase0, sampling_rate, \
                             freq_min, freq_max, freq_ref, phi_ref):
        """
        approximant -- model approximant
        amplitude0  -- amplitude pN order: -1 means include all
        phase0      -- phase pN order: -1 means include all
        sampling_rate    -- sampling rate in Hz
        segment_duration -- segment duration in sec
        freq_min -- start frequency in Hz
        freq_max -- end frequency in Hz
        freq_ref -- reference frequency for precessing spins in Hz
        phi_ref  -- final phase in degrees
        """
        
        self.approximant = GetApproximantFromString(approximant)
        self.sampling_rate = sampling_rate # Hz
        self.amplitude0 = amplitude0
        self.phase0 = phase0
        self.freq_min = freq_min # Hz, start frequency
        self.freq_max = freq_max # Hz, end frequency
        self.freq_ref = freq_ref # Hz, reference frequency for precessing spins
        self.phi_ref  = phi_ref  # final phase in degrees
        self.waveform_flags = SimInspiralCreateWaveformFlags()
        
    def time_domain(self, binary):
        """
        Compute time-domain template model of the gravitational wave for a given compact binary.
        Ref: http://software.ligo.org/docs/lalsuite/lalsimulation/group__lalsimulation__inspiral.html
        """
        
         # build structure containing variable with default values
        extra_params = lal.CreateDict()
        DictInsertREAL8Value(extra_params,"Lambda1", binary.lambda1)
        SimInspiralWaveformParamsInsertTidalLambda1(extra_params, binary.lambda1)
        DictInsertREAL8Value(extra_params,"Lambda2", binary.lambda2)
        SimInspiralWaveformParamsInsertTidalLambda2(extra_params, binary.lambda2)
        DictInsertINT4Value(extra_params, "amplitude0", self.amplitude0)
        DictInsertINT4Value(extra_params, "phase0", self.phase0)
        
        return SimInspiralTD(binary.mass1 * LAL_MSUN_SI, binary.mass2 * LAL_MSUN_SI, \
                             binary.spin1['x'], binary.spin1['y'], binary.spin1['z'], \
                             binary.spin2['x'], binary.spin2['y'], binary.spin2['z'], \
                             binary.distance * 1.0e6 * LAL_PC_SI, math.radians(binary.iota), \
                             math.radians(self.phi_ref), math.radians(binary.longAscNodes), \
                             binary.eccentricity, binary.meanPerAno, \
                             1.0 / self.sampling_rate, self.freq_min, self.freq_ref, \
                             extra_params, self.approximant)

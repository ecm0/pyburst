
from __future__ import division
import pycbc
import lal
import itertools
import numpy
from pycbc.detector import Detector
from pycbc.inject import InjectionSet
from pycbc.types import TimeSeries
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils as ligolw_utils

end_time = float(lal.GPSTimeNow())
mass1 = 30
mass2 = 30
distance = 1e6 * lal.PC_SI
taper = 'TAPER_STARTEND'

random = numpy.random.uniform
latitude = numpy.arccos(random(low=-1, high=1))
longitude = random(low=0, high=2 * lal.PI)
inclination = numpy.arccos(random(low=-1, high=1))
polarization = random(low=0, high=2 * lal.PI)

# create LIGOLW document
xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())

# create sim inspiral table, link it to document and fill it
sim_table = lsctables.New(lsctables.SimInspiralTable)
xmldoc.childNodes[-1].appendChild(sim_table)

row = sim_table.RowType()
row.waveform = 'SEOBNRv4_opt'
row.distance = distance
total_mass = mass1 + mass2
row.mass1 = mass1
row.mass2 = mass2
row.eta = mass1 * mass2 / total_mass ** 2
row.mchirp = total_mass * row.eta ** (3. / 5.)
row.latitude = latitude
row.longitude = longitude
row.inclination = inclination
row.polarization = polarization
row.phi0 = 0
row.f_lower = 20
row.f_final = lal.C_SI ** 3 / \
         (6. ** (3. / 2.) * lal.PI * lal.G_SI * total_mass)
row.spin1x = row.spin1y = row.spin1z = 0
row.spin2x = row.spin2y = row.spin2z = 0
row.alpha1 = 0
row.alpha2 = 0
row.alpha3 = 0
row.alpha4 = 0
row.alpha5 = 0
row.alpha6 = 0
row.alpha = 0
row.beta = 0
row.theta0 = 0
row.psi0 = 0
row.psi3 = 0
row.geocent_end_time = int(end_time)
row.geocent_end_time_ns = int(1e9 * (end_time - row.geocent_end_time))
row.end_time_gmst = lal.GreenwichMeanSiderealTime(
         lal.LIGOTimeGPS(end_time))
for d in 'lhvgt':
     row.__setattr__('eff_dist_' + d, row.distance)
     row.__setattr__(d + '_end_time', row.geocent_end_time)
     row.__setattr__(d + '_end_time_ns', row.geocent_end_time_ns)
row.amp_order = 0
row.coa_phase = 0
row.bandpass = 0
row.taper = inj.taper
row.numrel_mode_min = 0
row.numrel_mode_max = 0
row.numrel_data = None
row.source = 'ANTANI'
row.process_id = 'process:process_id:0'
row.simulation_id = 'sim_inspiral:simulation_id:0'

sim_table.append(row)

inj_file = open("injection.xml","w+")
ligolw_utils.write_fileobj(xmldoc, inj_file)

injection_set = InjectionSet("injection.xml")

sample_rate = 4096 # Hz
for det in [Detector(d) for d in ['H1', 'L1', 'V1']]:
         ts = TimeSeries(numpy.zeros(int(10 * sample_rate)),
                                 delta_t=1/sample_rate,
                                 epoch=lal.LIGOTimeGPS(end_time - 5),
                                 dtype=numpy.float64)

         injection_set.apply(ts, det.name)
         max_amp, max_loc = ts.abs_max_loc()
         pylab.plot(ts,label=det.name)

pylab.legend()

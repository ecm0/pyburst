
# test antenna_pattern

for d in network:
    print("{}:".format(d))
    print(d.antenna_pattern(*pt_eq.coords(fmt='lonlat', unit='radians'), 0, ref_time=time))
    print(d.antenna_pattern(*pt_geo.coords(fmt='lonlat', unit='radians'), 0, ref_time=None))

# Test delay
    
    print(d.time_delay_from_earth_center(*pt_eq.coords(fmt='lonlat', unit='radians'), ref_time=time))
    print(d.time_delay_from_earth_center(*pt_geo.coords(fmt='lonlat', unit='radians'), ref_time=None))


## Test project_strain

    
my_antenna_patt = numpy.array([d.antenna_pattern(*pt_eq.coords(fmt='lonlat', unit='radians'),0,ref_time=time) for d in network])
my_delays = numpy.array([d.time_delay_from_earth_center(*pt_eq.coords(fmt='lonlat', unit='radians'), ref_time=time) for d in network])

# ix = close_pixels[0][1]
# print('pixel is {}'.format(ix))
# print('antenna patt: {}'.format(antenna_patterns[ix][:,0]))
# print('delays: {}'.format(delays[ix]))
      
fig, axes = plt.subplots()
for h,d in zip(hoft,network):
    plt.plot(h, label=d)

plt.plot(time+1,0,'k+', markersize=10)
plt.gca().set_prop_cycle(None)
for t in my_delays:
# for t in delays[ix]:
    plt.plot(time+1+t,0,'o')
    
plt.gca().set_prop_cycle(None)
for a in my_antenna_patt[:,1]:
#for a in antenna_patterns[ix][:,0]:
    plt.plot([time, time+1],[a, a],'-')
    
plt.xlim([time+0.95,time+1.05])
plt.xlabel('time (s)')
plt.ylabel('')
plt.legend(loc='upper right')
plt.show()

# Test delay vs project_strain

deltat = []
for h in hoft:
    ix, = numpy.where(numpy.abs(h) > numpy.max(h)/10)
    t_end = h.t0.value + ix[-1]/sampling_rate
    deltat.append(t_end - (time+1))
print(numpy.diff(deltat), numpy.diff(my_delays))

# Test sky points in grid have similar antennna patterns and delays

close_pixels = healpy.pixelfunc.get_interp_weights(sky.nside,*pt_geo.coords(fmt='colatlon'),sky.order)
for ix in close_pixels[0]:
    print('ix={}: patterns: {}, delays: {}'.format(ix,antenna_patterns[ix],delays[ix]))

lon, lat = sky.grid.healpix_to_lonlat(range(sky.grid.npix))

fig, axes = plt.subplots()
for lo,la in zip(lon,lat):
    plt.plot(lo, la,'g.')
plt.plot(pt_geo.lon, pt_geo.lat,'rx')
for ix in close_pixels[0]:
    print(ix)
    plt.plot(lon[ix], lat[ix],'b.')

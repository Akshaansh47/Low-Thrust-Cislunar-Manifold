import spiceypy as sp
from skyfield.api import load
import numpy as np


#This is the skyfield implementation
ts = load.timescale()
eph = load('de440s.bsp')
earth = eph['Earth']
sun = eph['Sun']
moon = eph['Moon']
ts = load.timescale()
jd = 2460785.5000000
earthp = earth.at(ts.ut1_jd(jd)).position.km
sunp = sun.at(ts.ut1_jd(jd)).position.km
moonp = moon.at(ts.ut1_jd(jd)).position.km

earthv = earth.at(ts.ut1_jd(jd)).velocity.km_per_s
sunv = sun.at(ts.ut1_jd(jd)).velocity.km_per_s
moonv = moon.at(ts.ut1_jd(jd)).velocity.km_per_s


print(earthp-moonp)
print(sunp)
print(moonp)

#print(np.linalg.norm(earthp-moonp))

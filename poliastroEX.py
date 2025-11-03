from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import dynamics_equations

import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
r0 = [6578, 0, 0] * u.km
v0 = [0, 7.5, 0] * u.km / u.s
# orb = Orbit.from_vectors(Earth, r0, v0)

state = [6800 * u.km, 0.03 * u.one, 54 * u.deg, 360 * u.deg, 0 * u.deg, 0 * u.deg]
orb = Orbit.from_classical(Earth, *state)
print(dynamics_equations(
    0 * u.s, 
    np.hstack((orb.r.to(u.km).value, orb.v.to(u.km / u.s).value)),
    Earth.mass.value * u.G
    ))
# Time span for the simulation (keep period as a float in seconds)
period = orb.period.to(u.s).value
times = np.linspace(0, period * 500, num=300) * u.s  # Generate 300 time points
dt = times[1] - times[0]
# Create the propagator once and reuse it (avoid constructing inside loop)
propagator = CowellPropagator(f=dynamics_equations)

# Step through the orbit and record RAAN
a = [orb.a.to(u.km).value]
ecc = [orb.ecc.to(u.one).value]
inc = [orb.inc.to(u.deg).value]
raan = [orb.raan.to(u.deg).value]
w = [orb.argp.to(u.deg).value]
t = [orb.period.to(u.s).value / 60]  # Convert period to minutes

for _ in times[1:]:
    orb = orb.propagate(dt, method=propagator)
    a.append(orb.a.to(u.km).value)
    ecc.append(orb.ecc.to(u.one).value)
    inc.append(orb.inc.to(u.deg).value)
    raan.append(orb.raan.to(u.deg).value)
    w.append(orb.argp.to(u.deg).value)
    t.append(orb.period.to(u.s).value / 60)  # Convert period to minutes
    print(orb.epoch)

# Plotting the RAAN over time and save a PNG as fallback
fig = plt.figure()
ax = fig.add_subplot(3, 2, 1)
ax.plot(times.to(u.s).value, a, label="Semi-major Axis")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Semi-major Axis (km)")
ax.set_title("Satellite Semi-major Axis Over Time")
ax.grid()

ax = fig.add_subplot(3, 2, 2)
ax.plot(times.to(u.s).value, ecc, label="Eccentricity")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Eccentricity")
ax.set_title("Satellite Eccentricity Over Time")
ax.grid()

ax = fig.add_subplot(3, 2, 3)
ax.plot(times.to(u.s).value, inc, label="Inclination")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Inclination (deg)")
ax.set_title("Satellite Inclination Over Time")
ax.grid()

ax = fig.add_subplot(3, 2, 4)
ax.plot(times.to(u.s).value, raan, label="RAAN")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RAAN (deg)")
ax.set_title("Satellite RAAN Over Time")
ax.grid()

ax = fig.add_subplot(3, 2, 5)
ax.plot(times.to(u.s).value, w, label="Argument of Perigee")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Argument of Perigee (deg)")
ax.set_title("Satellite Argument of Perigee Over Time")
ax.grid()

ax = fig.add_subplot(3, 2, 6)
ax.plot(times.to(u.s).value, t, label="Period")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Period (min)")
ax.set_title("Satellite Period Over Time")
ax.grid()

plt.show()


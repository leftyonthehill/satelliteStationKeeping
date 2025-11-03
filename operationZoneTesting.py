from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import ref_dynamics_equations

import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
r0 = [6578, 0, 0] * u.km
v0 = [0, 7.5, 0] * u.km / u.s
# orb = Orbit.from_vectors(Earth, r0, v0)

state = [6600 * u.km, 0.03 * u.one, 54 * u.deg, 360 * u.deg, 0 * u.deg, 0 * u.deg]
orbForward = Orbit.from_classical(Earth, *state)
orbBackward = Orbit.from_classical(Earth, *state)
r, v = orbForward.rv()
ref_state = [[r[0].value], [r[1].value], [r[2].value]]
# Operation Bounding Zone parameters
altitude_bounds = 10 * u.km  # +/- from nominal altitude
timing_bounds = 1800 * u.s  # +/- from nominal orbital phasing
drift_bounds = 0.1 * u.deg  # radial drift limits


# Time span for the simulation (keep period as a float in seconds)
# Create the propagator once and reuse it (avoid constructing inside loop)
propagator = CowellPropagator(f=ref_dynamics_equations)

numSteps = 3
dt = timing_bounds / numSteps
print(dt)
for _ in range(numSteps):
    orbForward.propagate(dt, method=propagator)
    orbBackward.propagate(-dt, method=propagator)
    forward_r, forward_v = orbForward.rv()
    backward_r, backward_v = orbBackward.rv()
    print(orbForward.epoch)
    ref_state[0].append(forward_r[0].value)
    ref_state[1].append(forward_r[1].value)
    ref_state[2].append(forward_r[2].value)
    ref_state[0].insert(0, backward_r[0].value)
    ref_state[1].insert(0, backward_r[1].value)
    ref_state[2].insert(0, backward_r[2].value)

print(ref_state[1])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot(ref_state[0][:], ref_state[1][:], ref_state[2][:], label="Reference Orbit")
plt.show()
print("Operation Zone plotted.")


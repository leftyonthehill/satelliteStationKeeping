from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import ref_dynamics_equations, dynamics_equations
from op_zone_builder import op_zone_frame
from RIC_offset import RIC_offset
import numpy as np
import matplotlib.pyplot as plt

state = [6800 * u.km, 0.003 * u.one, 54 * u.deg, 360 * u.deg, 0 * u.deg, 0 * u.deg]
ref_orb = Orbit.from_classical(Earth, *state)
r, v = ref_orb.rv()
currentState = [r[0].value, r[1].value, r[2].value, v[0].value, v[1].value, v[2].value]
timeBound = 3 # seconds
radialBound = 10 # km
driftBound = 0.1 # degrees
ops_bounds = [
    timeBound, 
    radialBound, 
    driftBound
    ]  # [timing_bounds (s), altitude_bounds (km), drift_bounds (deg)]
trajectory, op_zone = op_zone_frame(currentState, ops_bounds)

warningScale = 4
warning_bounds = [
    timeBound * warningScale**(1/3), 
    radialBound * warningScale**(1/3), 
    driftBound * warningScale**(1/3)
]  # [timing_bounds (s), altitude_bounds (km), drift_bounds (deg)]
trajectory, warning_zone = op_zone_frame(currentState, warning_bounds)

maxBoundsScale = 32
max_bounds = [
    timeBound * maxBoundsScale**(1/3), 
    radialBound * maxBoundsScale**(1/3), 
    driftBound * maxBoundsScale**(1/3)
]  # [timing_bounds (s), altitude_bounds (km), drift_bounds (deg)]
trajectory, max_zone = op_zone_frame(currentState, max_bounds)

steps = 100
tof = ref_orb.period.to(u.s).value * 10  # Simulate for ten orbital periods
dt = tof / steps
truth_orb = Orbit.from_classical(Earth, *state)
pertubationDifferences = [[0], [0], [0]]
for _ in range(steps):
    truth_orb = truth_orb.propagate(dt << u.s, method=CowellPropagator(f=dynamics_equations))
    rTruth, vTruth = truth_orb.rv()
    stateTruth = [rTruth[0].value, rTruth[1].value, rTruth[2].value, vTruth[0].value, vTruth[1].value, vTruth[2].value]

    ref_orb = ref_orb.propagate(dt << u.s, method=CowellPropagator(f=ref_dynamics_equations))
    rRef, vRef = ref_orb.rv()
    stateRef = [rRef[0].value, rRef[1].value, rRef[2].value, vRef[0].value, vRef[1].value, vRef[2].value]

    ricOffset = RIC_offset(stateRef, stateTruth)
    pertubationDifferences[0].append(ricOffset[0])
    pertubationDifferences[1].append(ricOffset[1])
    pertubationDifferences[2].append(ricOffset[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[0], trajectory[1], trajectory[2], label="Reference Orbit Path", color='blue')
ax.plot(pertubationDifferences[0], pertubationDifferences[1], pertubationDifferences[2], label="Perturbation Difference", color='black', linestyle='dashed')

ax.plot(op_zone[0][0], op_zone[0][1], op_zone[0][2], label="Min Altitude, Below Plane", color='green')
ax.plot(op_zone[1][0], op_zone[1][1], op_zone[1][2], label="Min Altitude, Above Plane", color='green')
ax.plot(op_zone[2][0], op_zone[2][1], op_zone[2][2], label="Max Altitude, Below Plane", color='green')
ax.plot(op_zone[3][0], op_zone[3][1], op_zone[3][2], label="Max Altitude, Above Plane", color='green')

ax.plot(warning_zone[0][0], warning_zone[0][1], warning_zone[0][2], label="Min Altitude, Below Plane", color='orange')
ax.plot(warning_zone[1][0], warning_zone[1][1], warning_zone[1][2], label="Min Altitude, Above Plane", color='orange')
ax.plot(warning_zone[2][0], warning_zone[2][1], warning_zone[2][2], label="Max Altitude, Below Plane", color='orange')
ax.plot(warning_zone[3][0], warning_zone[3][1], warning_zone[3][2], label="Max Altitude, Above Plane", color='orange')

ax.plot(max_zone[0][0], max_zone[0][1], max_zone[0][2], label="Min Altitude, Below Plane", color='red')
ax.plot(max_zone[1][0], max_zone[1][1], max_zone[1][2], label="Min Altitude, Above Plane", color='red')
ax.plot(max_zone[2][0], max_zone[2][1], max_zone[2][2], label="Max Altitude, Below Plane", color='red')
ax.plot(max_zone[3][0], max_zone[3][1], max_zone[3][2], label="Max Altitude, Above Plane", color='red')
ax.set_xlabel("R (km)")
ax.set_ylabel("I (km)")
ax.set_zlabel("C (km)")
ax.set_title("Operation Zone Frame")

ax.set_aspect('equal')
plt.show()
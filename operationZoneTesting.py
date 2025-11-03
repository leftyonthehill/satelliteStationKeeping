from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import ref_dynamics_equations

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Initial conditions
    r0 = [6578, 0, 0] * u.km
    v0 = [0, 7.5, 0] * u.km / u.s
    # orb = Orbit.from_vectors(Earth, r0, v0)

    state = [6600 * u.km, 0.03 * u.one, 54 * u.deg, 360 * u.deg, 0 * u.deg, 0 * u.deg]
    orb = Orbit.from_classical(Earth, *state)
    ref_state = [orb.rv()]

    # Operation Bounding Zone parameters
    altitude_bounds = 10 * u.km  # +/- from nominal altitude
    timing_bounds = 15 * u.sec  # +/- from nominal orbital phasing
    drift_bounds = 0.1 * u.deg  # radial drift limits


    # Time span for the simulation (keep period as a float in seconds)
    # Create the propagator once and reuse it (avoid constructing inside loop)
    propagator = CowellPropagator(f=ref_dynamics_equations)

    numSteps = 50
    orbForward = orb.copy()
    orbBackward = orb.copy()
    for _ in range(numSteps):
        orbForward.propagate(timing_bounds / numSteps, method=propagator)
        orbBackward.propagate(-timing_bounds / numSteps, method=propagator)
        ref_state.append(orbForward.rv())
        ref_state.insert(0, orbBackward.rv())
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(ref_state[:][0], ref_state[:][1], ref_state[:][2], label="Reference Orbit")

    
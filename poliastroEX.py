from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import dynamics_equations

import numpy as np
import matplotlib
# Try to select a GUI backend that opens a window on most systems (Windows: TkAgg)
try:
    matplotlib.use("TkAgg")
except Exception:
    # If backend selection fails, we'll still proceed and save the figure to a file as a fallback
    pass
import matplotlib.pyplot as plt

def main():
    plt.close('all')

    # Initial conditions
    r0 = [6750, 0, 0] * u.km
    v0 = [0, 7.5, 0] * u.km / u.s
    # orb = Orbit.from_vectors(Earth, r0, v0)

    state = [6750 * u.km, 0.03 * u.one, 54 * u.deg, 360 * u.deg, 0 * u.deg, 0 * u.deg]
    orb = Orbit.from_classical(Earth, *state)

    # Time span for the simulation (keep period as a float in seconds)
    period = orb.period.to(u.s).value
    times = np.linspace(0, period * 10, num=1000) * u.s  # Generate 1000 time points
    dt = times[1] - times[0]

    # Create the propagator once and reuse it (avoid constructing inside loop)
    propagator = CowellPropagator(f=dynamics_equations)

    # Step through the orbit and record RAAN
    raan = [orb.raan.to(u.deg).value]
    for _ in times[1:]:
        orb = orb.propagate(dt, method=propagator)
        raan.append(orb.raan.to(u.deg).value)

    # Plotting the RAAN over time and save a PNG as fallback
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times.to(u.s).value, raan, label="RAAN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RAAN (deg)")
    ax.set_title("Satellite RAAN Over Time")
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()


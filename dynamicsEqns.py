from poliastro.twobody import Orbit
from poliastro.bodies import Earth

from poliastro.core.propagation import func_twobody
from poliastro.core.perturbations import J2_perturbation

from astropy import units as u

import numpy as np

def dynamics_equations(t0, state_vector, mu):
    twoBodyAccel = func_twobody(t0, state_vector, mu)
    j2Perturbation = J2_perturbation(t0, state_vector, mu, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
    dragAcceleration = atmDrag(state_vector, beta=30)  # Example beta value
    # func_twobody returns a 6-element derivative [v, a]; J2_perturbation returns a 3-element
    # acceleration vector. Add the J2 acceleration to the last three entries (accelerations)
    # and return the full 6-element derivative.
    twoBodyAccel[3:6] = twoBodyAccel[3:6] + j2Perturbation + dragAcceleration
    return twoBodyAccel

def atmDrag(state_vector, beta):
    """Compute acceleration due to atmospheric drag.

    Parameters
    ----------
    state_vector : array_like
        State vector [x, y, z, vx, vy, vz] in km and km/s.
    beta : float
        Ballistic coefficient in kg/m^2.

    Returns
    -------
    ndarray
        Acceleration vector due to atmospheric drag in km/s^2.
    """
    earthRotationRate = 7.2921159e-5  # rad/s

    # Compute the effective velocity by accounting for Earth's rotation
    effective_velocity = state_vector[3:6] - np.cross([0, 0, earthRotationRate], state_vector[:3])

    # Placeholder for atmospheric density model
    rho = 5e-10  # kg/km^3, example constant density

    v_rel = effective_velocity  # Relative velocity vector in km/s
    v_rel_mag = np.linalg.norm(v_rel)

    # Drag acceleration formula: a_drag = -0.5 * (rho * v^2 / beta) * (v / |v|)
    a_drag = -0.5 * (rho * v_rel_mag**2 / beta) * (v_rel / v_rel_mag)

    return a_drag
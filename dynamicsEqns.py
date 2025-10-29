from poliastro.twobody import Orbit
from poliastro.bodies import Earth

from poliastro.core.propagation import func_twobody
from poliastro.core.perturbations import J2_perturbation

from astropy import units as u

def dynamics_equations(t0, state_vector, mu):
    twoBodyAccel = func_twobody(t0, state_vector, mu)
    j2Perturbation = J2_perturbation(t0, state_vector, mu, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
    # func_twobody returns a 6-element derivative [v, a]; J2_perturbation returns a 3-element
    # acceleration vector. Add the J2 acceleration to the last three entries (accelerations)
    # and return the full 6-element derivative.
    twoBodyAccel = twoBodyAccel.copy()
    twoBodyAccel[3:6] = twoBodyAccel[3:6] + j2Perturbation
    return twoBodyAccel
from astropy import units as u
from poliastro.twobody import Orbit
from control import lqr
import numpy as np

def CW_LQR(orb: Orbit, Q=np.diag([1e3, 1e3, 1e3, 1e1, 1e1, 1e1]), R=np.eye(3) * 1e6):
    """Computes the LQR gain matrix for the Clohessy-Wiltshire equations.

    Parameters
    ----------
    orb : Orbit
        The reference orbit around which to linearize.
    Q : ndarray
        State weighting matrix.
    R : ndarray
        Control weighting matrix.

    Returns
    -------
    K : ndarray
        The LQR gain matrix.
    """
    # Mean motion
    n = orb.n.to_value(u.rad / u.s)

    # State-space representation of CW equations
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [3*n**2, 0, 0, 0, 2*n, 0],
                  [0, 0, 0, -2*n, 0, 0],
                  [0, 0, -n**2, 0, 0, 0]])

    B = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    # Compute LQR gain matrix
    K, _, _ = lqr(A, B, Q, R)

    return K
    
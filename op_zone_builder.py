import numpy as np
from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import ref_dynamics_equations
import matplotlib.pyplot as plt

def op_zone_frame(truthState, ops_bounds):
    """Generates operation zone frame based on reference state and operation bounds."""
    dt = ops_bounds[0]
    dr = ops_bounds[1]
    dB = ops_bounds[2]

    r_ref = np.array(truthState[:3])
    v_ref = np.array(truthState[3:6])

    # Compute unit vectors for RIC frame
    R = r_ref / np.linalg.norm(r_ref)
    h_vec = np.cross(r_ref, v_ref)
    C = h_vec / np.linalg.norm(h_vec)
    I = np.cross(C, R)

    # Rotation matrix from inertial to RIC
    rotMatrix = np.vstack((R, I, C))

    propagator = CowellPropagator(f=ref_dynamics_equations)
    steps = 50

    trajectory = [[0], [0], [0]]
    op_zone_points = [
        [[],[],[]], # min altitude, below orbital plane
        [[],[],[]], # min altitude, above orbital plane
        [[],[],[]], # max altitude, below orbital plane
        [[],[],[]]  # max altitude, above orbital plane
    ]

    # point 0
    minAlt = r_ref - dr * R
    maxAlt = r_ref + dr * R

    lambda_Matrix = np.vstack(([0, -I[2], I[1]],
                            [I[2], 0, -I[0]],
                            [-I[1], I[0], 0]))
    minAlt_belowPlane = (np.eye(3) + np.sin(np.deg2rad(dB)) * lambda_Matrix +
                        (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ minAlt
    minAlt_belowPlane = rotMatrix @ (minAlt_belowPlane - r_ref)
    op_zone_points[0][0].append(minAlt_belowPlane[0])
    op_zone_points[0][1].append(minAlt_belowPlane[1])
    op_zone_points[0][2].append(minAlt_belowPlane[2])

    minAlt_abovePlane = (np.eye(3) - np.sin(np.deg2rad(dB)) * lambda_Matrix +
                        (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ minAlt
    minAlt_abovePlane = rotMatrix @ (minAlt_abovePlane - r_ref)
    op_zone_points[1][0].append(minAlt_abovePlane[0])
    op_zone_points[1][1].append(minAlt_abovePlane[1])
    op_zone_points[1][2].append(minAlt_abovePlane[2])

    maxAlt_belowPlane = (np.eye(3) + np.sin(np.deg2rad(dB)) * lambda_Matrix +
                        (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ maxAlt
    maxAlt_belowPlane = rotMatrix @ (maxAlt_belowPlane - r_ref)
    op_zone_points[2][0].append(maxAlt_belowPlane[0])
    op_zone_points[2][1].append(maxAlt_belowPlane[1])
    op_zone_points[2][2].append(maxAlt_belowPlane[2])

    maxAlt_abovePlane = (np.eye(3) - np.sin(np.deg2rad(dB)) * lambda_Matrix +
                        (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ maxAlt
    maxAlt_abovePlane = rotMatrix @ (maxAlt_abovePlane - r_ref)
    op_zone_points[3][0].append(maxAlt_abovePlane[0])
    op_zone_points[3][1].append(maxAlt_abovePlane[1])
    op_zone_points[3][2].append(maxAlt_abovePlane[2])

    forwards = Orbit.from_vectors(Earth, r_ref * u.km, v_ref * u.km / u.s)
    backwards = Orbit.from_vectors(Earth, r_ref * u.km, v_ref * u.km / u.s)
    for _ in range(steps):
        forwards = forwards.propagate(dt / steps << u.s, method=propagator)
        r, v = forwards.rv()
        ricOffset = rotMatrix @ (r.value - r_ref)
        trajectory[0].append(ricOffset[0])
        trajectory[1].append(ricOffset[1])
        trajectory[2].append(ricOffset[2])

        R_prop = r / np.linalg.norm(r)
        h_vec = np.cross(r, v)
        C_prop = h_vec / np.linalg.norm(h_vec)
        I_prop = np.cross(C_prop, R_prop)
        
        minAlt = r.value - dr * R_prop
        maxAlt = r.value + dr * R_prop
        lambda_Matrix = np.vstack(([0, -I_prop[2], I_prop[1]],
                                [I_prop[2], 0, -I_prop[0]],
                                [-I_prop[1], I_prop[0], 0]))
        
        minAlt_belowPlane = (np.eye(3) + np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ minAlt
        minAlt_belowPlane = rotMatrix @ (minAlt_belowPlane - r_ref)
        op_zone_points[0][0].append(minAlt_belowPlane[0])
        op_zone_points[0][1].append(minAlt_belowPlane[1])
        op_zone_points[0][2].append(minAlt_belowPlane[2])

        minAlt_abovePlane = (np.eye(3) - np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ minAlt
        minAlt_abovePlane = rotMatrix @ (minAlt_abovePlane - r_ref)
        op_zone_points[1][0].append(minAlt_abovePlane[0])
        op_zone_points[1][1].append(minAlt_abovePlane[1])
        op_zone_points[1][2].append(minAlt_abovePlane[2])

        maxAlt_belowPlane = (np.eye(3) + np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ maxAlt
        maxAlt_belowPlane = rotMatrix @ (maxAlt_belowPlane - r_ref)
        op_zone_points[2][0].append(maxAlt_belowPlane[0])
        op_zone_points[2][1].append(maxAlt_belowPlane[1])
        op_zone_points[2][2].append(maxAlt_belowPlane[2])

        maxAlt_abovePlane = (np.eye(3) - np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ maxAlt
        maxAlt_abovePlane = rotMatrix @ (maxAlt_abovePlane - r_ref)
        op_zone_points[3][0].append(maxAlt_abovePlane[0])
        op_zone_points[3][1].append(maxAlt_abovePlane[1])
        op_zone_points[3][2].append(maxAlt_abovePlane[2])

        backwards = backwards.propagate(-dt / steps << u.s, method=propagator) 
        r, v = backwards.rv()
        ricOffset = rotMatrix @ (r.value - r_ref)
        trajectory[0].insert(0, ricOffset[0])
        trajectory[1].insert(0, ricOffset[1])
        trajectory[2].insert(0, ricOffset[2])

        R_prop = r / np.linalg.norm(r)
        h_vec = np.cross(r, v)
        C_prop = h_vec / np.linalg.norm(h_vec)
        I_prop = np.cross(C_prop, R_prop)

        minAlt = r.value - dr * R_prop
        maxAlt = r.value + dr * R_prop
        lambda_Matrix = np.vstack(([0, -I_prop[2], I_prop[1]],
                                [I_prop[2], 0, -I_prop[0]],
                                [-I_prop[1], I_prop[0], 0]))
        
        minAlt_belowPlane = (np.eye(3) + np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ minAlt
        minAlt_belowPlane = rotMatrix @ (minAlt_belowPlane - r_ref)
        op_zone_points[0][0].insert(0, minAlt_belowPlane[0])
        op_zone_points[0][1].insert(0, minAlt_belowPlane[1])
        op_zone_points[0][2].insert(0, minAlt_belowPlane[2])

        minAlt_abovePlane = (np.eye(3) - np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ minAlt
        minAlt_abovePlane = rotMatrix @ (minAlt_abovePlane - r_ref)
        op_zone_points[1][0].insert(0, minAlt_abovePlane[0])
        op_zone_points[1][1].insert(0, minAlt_abovePlane[1])
        op_zone_points[1][2].insert(0, minAlt_abovePlane[2])

        maxAlt_belowPlane = (np.eye(3) + np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ maxAlt
        maxAlt_belowPlane = rotMatrix @ (maxAlt_belowPlane - r_ref)
        op_zone_points[2][0].insert(0, maxAlt_belowPlane[0])
        op_zone_points[2][1].insert(0, maxAlt_belowPlane[1])
        op_zone_points[2][2].insert(0, maxAlt_belowPlane[2])

        maxAlt_abovePlane = (np.eye(3) - np.sin(np.deg2rad(dB)) * lambda_Matrix +
                            (1 - np.cos(np.deg2rad(dB))) * (lambda_Matrix @ lambda_Matrix)) @ maxAlt
        maxAlt_abovePlane = rotMatrix @ (maxAlt_abovePlane - r_ref)
        op_zone_points[3][0].insert(0, maxAlt_abovePlane[0])
        op_zone_points[3][1].insert(0, maxAlt_abovePlane[1])
        op_zone_points[3][2].insert(0, maxAlt_abovePlane[2])

    op_zone_points[0][0].insert(0, op_zone_points[1][0][0])
    op_zone_points[0][1].insert(0, op_zone_points[1][1][0])
    op_zone_points[0][2].insert(0, op_zone_points[1][2][0])
    op_zone_points[0][0].append(op_zone_points[1][0][-1])
    op_zone_points[0][1].append(op_zone_points[1][1][-1])
    op_zone_points[0][2].append(op_zone_points[1][2][-1])

    op_zone_points[1][0].insert(0, op_zone_points[3][0][0])
    op_zone_points[1][1].insert(0, op_zone_points[3][1][0])
    op_zone_points[1][2].insert(0, op_zone_points[3][2][0])
    op_zone_points[1][0].append(op_zone_points[3][0][-1])
    op_zone_points[1][1].append(op_zone_points[3][1][-1])
    op_zone_points[1][2].append(op_zone_points[3][2][-1])

    op_zone_points[2][0].insert(0, op_zone_points[0][0][1])
    op_zone_points[2][1].insert(0, op_zone_points[0][1][1])
    op_zone_points[2][2].insert(0, op_zone_points[0][2][1])
    op_zone_points[2][0].append(op_zone_points[0][0][-2])
    op_zone_points[2][1].append(op_zone_points[0][1][-2])
    op_zone_points[2][2].append(op_zone_points[0][2][-2])

    op_zone_points[3][0].insert(0, op_zone_points[2][0][1])
    op_zone_points[3][1].insert(0, op_zone_points[2][1][1])
    op_zone_points[3][2].insert(0, op_zone_points[2][2][1])
    op_zone_points[3][0].append(op_zone_points[2][0][-2])
    op_zone_points[3][1].append(op_zone_points[2][1][-2])
    op_zone_points[3][2].append(op_zone_points[2][2][-2])

    return trajectory, op_zone_points

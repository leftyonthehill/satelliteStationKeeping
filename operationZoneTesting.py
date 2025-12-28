from astropy import units as u
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import ref_dynamics_equations, dynamics_equations
from op_zone_builder import op_zone_frame
from op_zone_check import op_zone_check
from RIC_offset import RIC_offset
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # registers 3D projection

state = [6887.0 * u.km, 0.003 * u.one, 54 * u.deg, 360 * u.deg, 0 * u.deg, 0 * u.deg]
ref_orb = Orbit.from_classical(Earth, *state)
r, v = ref_orb.rv()

currentState = [r[0].value, r[1].value, r[2].value, v[0].value, v[1].value, v[2].value]
timeBound = 1 # seconds ## 3
radialBound = 5 # km ## 10
driftBound = 0.05 # degrees ## 0.1
ops_bounds = [
    timeBound, 
    radialBound, 
    driftBound
]  # [timing_bounds (s), altitude_bounds (km), drift_bounds (deg)]
trajectory, op_zone = op_zone_frame(currentState, ops_bounds)

operationBoundaries = {
    "Operation Zone": op_zone
}

steps = 300
tof = round(1 / ref_orb.period.to(u.day).value) * ref_orb.period.to(u.s).value
dt = tof / steps
truth_orb = Orbit.from_classical(Earth, *state)
pertubationDifferences = [[0], [0], [0]]
for _ in range(steps):
    truth_orb = truth_orb.propagate(dt << u.s, method=CowellPropagator(f=dynamics_equations))
    rTruth, vTruth = truth_orb.rv()
    stateTruth = [rTruth[0].value, rTruth[1].value, rTruth[2].value, vTruth[0].value, vTruth[1].value, vTruth[2].value]

    if np.linalg.norm(stateTruth[0:3]) < Earth.R.to(u.km).value + 100:
        break  # Stop simulation if the satellite re-enters the atmosphere

    ref_orb = ref_orb.propagate(dt << u.s, method=CowellPropagator(f=ref_dynamics_equations))
    rRef, vRef = ref_orb.rv()
    stateRef = [rRef[0].value, rRef[1].value, rRef[2].value, vRef[0].value, vRef[1].value, vRef[2].value]

    ricOffset, _ = RIC_offset(stateRef, stateTruth)
    pertubationDifferences[0].append(ricOffset[0])
    pertubationDifferences[1].append(ricOffset[1])
    pertubationDifferences[2].append(ricOffset[2])
    
    op_zone_check(ricOffset, operationBoundaries)

"""
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
"""

# ---- Prepare data ------------------------------------------------
R_ref = np.array(trajectory)          # (3, N)  reference orbit in ECI
R_ric = np.array(pertubationDifferences)   # (3, N)  RIC offset of secondary

# ---- Figure ------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('R  (km)')
ax.set_ylabel('I  (km)')
ax.set_zlabel('C  (km)')
ax.set_title('RIC separation – drag makes secondary drift +I')
ax.set_aspect('equal')

# Limits – grow a little beyond the final separation
margin = 30  # km
ax.set_xlim(-max(abs(R_ric[0])) - margin, max(abs(R_ric[0])) + margin)
ax.set_ylim(-margin, max(R_ric[1]) + margin)          # I grows positive
ax.set_zlim(-max(abs(R_ric[2])) - margin, max(abs(R_ric[2])) + margin)

# ---- Artists that will be updated each frame --------------------
line_ref,  = ax.plot([], [], [], 'b-', lw=1.5, label='Reference (primary)')
line_sec,  = ax.plot([], [], [], 'k-', lw=1.5, label='Secondary (truth)')
point_ref, = ax.plot([], [], [], 'bo', markersize=6)
point_sec, = ax.plot([], [], [], 'ko', markersize=6)

# ---- Draw the operation zones once (static) ----------
def draw_zone(zone, color, alpha=0.4):
    for seg in zone:
        ax.plot(seg[0], seg[1], seg[2], color=color, alpha=alpha)
draw_zone(operationBoundaries['Operation Zone'],      'g')
ax.legend()

# --------------------------------------------------------------
# 3. Animation functions
# --------------------------------------------------------------
def init():
    line_ref.set_data([], [])
    line_ref.set_3d_properties([])
    line_sec.set_data([], [])
    line_sec.set_3d_properties([])
    point_ref.set_data([], [])
    point_ref.set_3d_properties([])
    point_sec.set_data([], [])
    point_sec.set_3d_properties([])
    return line_ref, line_sec, point_ref, point_sec

def animate(frame):
    # frame goes from 0 … N-1
    # Reference orbit (ECI) – convert to RIC for consistency
    # (the reference is always at (0,0,0) in RIC, so we just plot a point)
    point_ref.set_data([0], [0])
    point_ref.set_3d_properties([0])

    # Secondary in RIC
    line_sec.set_data(R_ric[0, :frame+1], R_ric[1, :frame+1])   # R, I
    line_sec.set_3d_properties(R_ric[2, :frame+1])              # C
    point_sec.set_data([R_ric[0, frame]], [R_ric[1, frame]])
    point_sec.set_3d_properties([R_ric[2, frame]])

    # Reference orbit trace (optional – show a short segment)
    # seg = max(0, frame-200)          # last 200 points ≈ 1 orbit
    # line_ref.set_data(R_ref[1, seg:frame+1], R_ref[0, seg:frame+1])
    # line_ref.set_3d_properties(R_ref[2, seg:frame+1])

    ax.set_title(f'RIC separation – step {frame} / {R_ric.shape[1]-1}')
    return line_ref, line_sec, point_ref, point_sec

# --------------------------------------------------------------
# 4. Build & save / show
# --------------------------------------------------------------
ani = animation.FuncAnimation(fig, animate, frames=R_ric.shape[1],
                              init_func=init, blit=False, interval=30)

# ---- Choose one of the two lines below ----
# ani.save('ric_drag_separation.mp4', writer='ffmpeg', fps=30, dpi=200)   # MP4
plt.show()                                                             # interactive
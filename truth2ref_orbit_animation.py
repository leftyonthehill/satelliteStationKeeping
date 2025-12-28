import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # registers 3D projection

def orbitAnimation(opsBoundaries, orbitHistory):
    # ---- Prepare data ------------------------------------------------
    animationStates = {
        "State Duration": [], # number of frames each state lasts
        "State Counter": 0    # counts which state we're in
    }

    # R_ric = [[],[],[]]
    R_ric = np.array([state["RIC_Offset"] for state in orbitHistory])
    R = R_ric[:,0]
    I = R_ric[:,1]
    C = R_ric[:,2]

    lastState = ""
    currentState = ""
    """for state in orbitHistory:
        R_ric[0].append(state["RIC_Offset"][0])
        R_ric[1].append(state["RIC_Offset"][1])
        R_ric[2].append(state["RIC_Offset"][2])

        currentState = state["OrbitStatus"]
        if currentState != lastState:
            stateKey = "State " + str(animationStates.keys().__len__() - 1)
            animationStates[stateKey] = [state["RIC_Offset"]]
            animationStates["State Duration"].append(1)
        else:
            stateKey = "State " + str(animationStates.keys().__len__() - 2)
            animationStates[stateKey].append(state["RIC_Offset"])
            animationStates["State Duration"][-1] += 1 """


    # ---- Figure ------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('R  (km)')
    ax.set_ylabel('I  (km)')
    ax.set_zlabel('C  (km)')
    ax.set_title('RIC separation animation')
    ax.set_aspect('equal')

    # Limits – grow a little beyond the final separation
    margin = 30  # km
    ax.set_xlim(-1 * np.max(np.abs(R)) - margin, np.max(np.abs(R)) + margin)
    ax.set_ylim(-margin, np.max(I) + margin)
    ax.set_zlim(-np.max(np.abs(C)) - margin, np.max(np.abs(C)) + margin)

    # ---- Artists that will be updated each frame --------------------
    line_coast,  = ax.plot([], [], [], 'k-', lw=1.5, label='Truth Offset')
    line_thrust,  = ax.plot([], [], [], 'r-', lw=1.5, label='Thrusting Correction')
    point_ref, = ax.plot([], [], [], 'bo', markersize=6)
    point_coast, = ax.plot([], [], [], 'ko', markersize=6)
    point_thrust, = ax.plot([], [], [], 'ro', markersize=6)

    # ---- Draw the operation zones once (static) ----------
    def draw_zone(zone, color, alpha=0.4):
        for seg in zone:
            ax.plot(seg[0], seg[1], seg[2], color=color, alpha=alpha)
    draw_zone(opsBoundaries,      'g')

    ax.legend()

    # --------------------------------------------------------------
    # 3. Animation functions
    # --------------------------------------------------------------
    def init():
        line_coast.set_data([], [])
        line_coast.set_3d_properties([])
        line_thrust.set_data([], [])
        line_thrust.set_3d_properties([])
        point_ref.set_data([], [])
        point_ref.set_3d_properties([])
        point_coast.set_data([], [])
        point_coast.set_3d_properties([])
        point_thrust.set_data([], [])
        point_thrust.set_3d_properties([])
        return line_coast, line_thrust, point_ref, point_coast, point_thrust

    def animate(frame):
        # frame goes from 0 … N-1
        # Reference orbit (ECI) – convert to RIC for consistency
        # (the reference is always at (0,0,0) in RIC, so we just plot a point)
        point_ref.set_data([0], [0])
        point_ref.set_3d_properties([0])
        line_coast.set_data(R[:frame+1], I[:frame+1])   # R, I
        line_coast.set_3d_properties(C[:frame+1])              # C
        point_coast.set_data([R[frame]], [I[frame]])
        point_coast.set_3d_properties([C[frame]])

        # Depending on the orbit status, plot in different styles
        if orbitHistory[frame]["OrbitStatus"] == "Out of Zone":
                line_thrust.set_data(R[:frame+1], I[:frame+1])   # R, I
                line_thrust.set_3d_properties(C[:frame+1])              # C
        else:
            line_thrust.set_data([], [])
            line_thrust.set_3d_properties([])

        ax.set_title(f'RIC separation - step {frame} / {len(R)-1}')
        return point_ref, line_coast, point_coast

    # --------------------------------------------------------------
    # 4. Build & save / show
    # --------------------------------------------------------------
    ani = animation.FuncAnimation(fig, animate, frames=len(orbitHistory),
                                init_func=init, blit=False, interval=60)

    # ---- Choose one of the two lines below ----
    # ani.save('ric_drag_separation.mp4', writer='ffmpeg', fps=30, dpi=200)   # MP4
    plt.show()   
    return ani                                                          # interactive
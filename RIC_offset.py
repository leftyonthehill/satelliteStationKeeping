def RIC_offset(refState, offsetState):
    """Converts state vector from inertial frame to RIC frame based on reference state."""
    import numpy as np

    r_ref = np.array(refState[0:3])
    v_ref = np.array(refState[3:6])
    r_offset = np.array(offsetState[0:3])

    # Compute unit vectors for RIC frame
    R = r_ref / np.linalg.norm(r_ref)
    h_vec = np.cross(r_ref, v_ref)
    C = h_vec / np.linalg.norm(h_vec)
    I = np.cross(C, R)

    # Rotation matrix from inertial to RIC
    rotMatrix = np.vstack((R, I, C))
    
    # Position in RIC frame
    delta_r = r_offset - r_ref
    r_RIC = rotMatrix.T @ delta_r

    return r_RIC

def main():
    state = [6578, 0, 0, 0, 7.5, 0]
    print(RIC_offset(state, [6600, 400, -900, 0, 7.5, 0]))

if __name__ == "__main__":
    main()
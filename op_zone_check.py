def is_within_bounds(state, bounds):
    """Check if the given state is within the specified operation zone bounds.

    Parameters
    ----------
    state : array_like
        State vector [R, I, C] in km.
    bounds : list
        Operation zone bounds 
        [
            [
                [R1_1, ..., R1_N], 
                [I1_1, ..., I1_N], 
                [C1_1, ..., C1_N]
            ], 
            [
                [R2_1, ..., R2_N], 
                [I2_1, ..., I2_N],
                [C2_1, ..., C2_N]
            ],
            [
                [R3_1, ..., R3_N], 
                [I3_1, ..., I3_N],
                [C3_1, ..., C3_N]
            ],
            [
                [R4_1, ..., R4_N], 
                [I4_1, ..., I4_N],
                [C4_1, ..., C4_N]
            ]
        ] in km.

    Returns
    -------
    bool
        True if the state is within bounds, False otherwise.
    """

    R, I, C = state
    minR, maxR = [], []
    minI, maxI = [], []
    minC, maxC = [], []
    for zone in bounds:
        R_bounds = zone[0]
        I_bounds = zone[1]
        C_bounds = zone[2]

        R_min, R_max = min(R_bounds), max(R_bounds)
        I_min, I_max = min(I_bounds), max(I_bounds)
        C_min, C_max = min(C_bounds), max(C_bounds)

        minR.append(R_min)
        maxR.append(R_max)
        minI.append(I_min)
        maxI.append(I_max)
        minC.append(C_min)
        maxC.append(C_max)

    if min(minR) <= R <= max(maxR) and min(minI) <= I <= max(maxI) and min(minC) <= C <= max(maxC):
        return True
    else:
        return False

def op_zone_check(state, op_zone_bounds: dict):
    """Check if the given state is within the operation zone bounds.

    Parameters
    ----------
    state : array_like
        State vector [R, I, C] in km.
    op_zone_bounds : dict
        Boundary Name: List of Bounds

    Returns
    -------
    bool
        True if the state is within the operation zone, False otherwise.
    """
    currentZone = ""
    for zone_name, zone_bounds in op_zone_bounds.items():
        if is_within_bounds(state, zone_bounds):
            print(f"Satellite is within its {zone_name} ({state}).")
            return zone_name
        else:
            currentZone = zone_name
    print(f"Satellite is outside its {currentZone} ({state}).")
    return currentZone
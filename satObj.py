from astropy import units as u
from astropy.time import Time
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from poliastro.twobody.propagation import CowellPropagator
from dynamicsEqns import ref_dynamics_equations, dynamics_equations
from op_zone_builder import op_zone_frame
from op_zone_check import is_within_bounds
from RIC_offset import RIC_offset
from CW_LQR_gain import CW_LQR

import numpy as np

class satellite():
    def __init__(self):
        self.name = "Generic Satellite"

        # Satellite physical properties
        self.beta = 15 * u.kg / u.m**2  # Ballistic coefficient
        self.agom = 1e-4 * u.m**2 / u.kg  # Area-gamma-to-mass ratio
        self.mass = 500 * u.kg  # Mass
        self.maxThrust = .017 * u.N  # Maximum thrust
        self.deltaV_consumed = []

        # Satellite Station keeping control properties
        self.Q = np.diag([5, 10, 1, 1, 5, 0.1])
        self.R = np.eye(3) * 1e7

        # Initial orbit
        self.epoch = Time.now()
        state = [6887.0 * u.km, 0.003 * u.one, 54 * u.deg, 360 * u.deg, 0 * u.deg, 0 * u.deg]
        self.refOrbit = Orbit.from_classical(Earth, *state)
        self.truthOrbit = Orbit.from_classical(Earth, *state)
        self.OrbitStatus = "Nominal"

        # Satellite operation zone parameters
        self.timeBound = 1  # seconds
        self.radialBound = 5  # km
        self.driftBound = 0.05  # degrees

        ops_bounds = [
            self.timeBound, 
            self.radialBound, 
            self.driftBound
        ]  # [timing_bounds (s), altitude_bounds (km), drift_bounds (deg)]
        r, v = self.refOrbit.rv()
        _, op_zone = op_zone_frame([*r.value, *v.value], ops_bounds)

        self.operationBoundaries = {
            "Operation Zone": op_zone
        }

        self.OrbitHistory = [{
            "Epoch": self.epoch,
            "RIC_Offset": [0.0, 0.0, 0.0],
            "OrbitStatus": self.OrbitStatus
        }]

    def getDeltaVConsumed(self):
        return self.deltaV_consumed

    def getTruthOrbit(self):
        return self.truthOrbit

    def getTruthState(self):
        rTruth, vTruth = self.truthOrbit.rv()
        return [rTruth[0].value, rTruth[1].value, rTruth[2].value, vTruth[0].value, vTruth[1].value, vTruth[2].value]
    
    def getRefOrbit(self):
        return self.refOrbit

    def getRefState(self):
        rRef, vRef = self.refOrbit.rv()
        return [rRef[0].value, rRef[1].value, rRef[2].value, vRef[0].value, vRef[1].value, vRef[2].value]
    
    def getOpsFrame(self):
        return self.operationBoundaries["Operation Zone"]
    
    def getOrbitHistory(self):
        return self.OrbitHistory

    def clearOrbitHistory(self):
        self.OrbitHistory = self.OrbitHistory[-1]

    def propagate(self, dt):
        self.epoch += dt << u.s
        
        ricOffset = []

        match self.OrbitStatus:
            case "Nominal":
                self.refOrbit = self.refOrbit.propagate(dt << u.s, method=CowellPropagator(f=ref_dynamics_equations))
                rREF, vREF = self.refOrbit.rv()

                self.truthOrbit = self.truthOrbit.propagate(dt << u.s, method=CowellPropagator(f=dynamics_equations))
                rTRUTH, vTRUTH = self.truthOrbit.rv()
                
                ricOffset = RIC_offset([*rREF.value, *vREF.value], [*rTRUTH.value, *vTRUTH.value])[0]

                # if not is_within_bounds(ricOffset, self.operationBoundaries["Operation Zone"]):
                exit_distance_threshold = 1.4 * self.radialBound
                if np.linalg.norm(ricOffset) > exit_distance_threshold:
                    self.OrbitStatus = "Correction Planning"

            case "Correction Planning":
                # Simulated delay for planning and thruster warm-up
                self.refOrbit = self.refOrbit.propagate(dt << u.s, method=CowellPropagator(f=ref_dynamics_equations))
                rREF, vREF = self.refOrbit.rv()
        
                self.truthOrbit = self.truthOrbit.propagate(dt << u.s, method=CowellPropagator(f=dynamics_equations))
                rTRUTH, vTRUTH = self.truthOrbit.rv()
                
                ricOffset = RIC_offset([*rREF.value, *vREF.value], [*rTRUTH.value, *vTRUTH.value])[0]
                self.deltaV_consumed.append(0.0)  # No delta-V consumed during planning

                self.OrbitStatus = "Out of Zone"
            case "Out of Zone":
                rREF, vREF = self.refOrbit.rv()
                rTRUTH, vTRUTH = self.truthOrbit.rv()
                ricPos, ricVel, ricRot = RIC_offset([*rREF.value, *vREF.value], [*rTRUTH.value, *vTRUTH.value])
                
                self.refOrbit = self.refOrbit.propagate(dt << u.s, method=CowellPropagator(f=ref_dynamics_equations))
                rREF, vREF = self.refOrbit.rv()

                K = CW_LQR(self.refOrbit ,self.Q, self.R)
                
                ricAccel = K @ np.array([*ricPos, *ricVel])  # km/s^2

                # maxAcceleration = self.maxThrust.value / 1000 / self.mass.value  # km/s^2
                thrustAcceleration = np.array([*vTRUTH.value]) / np.linalg.norm([*vTRUTH.value]) * self.maxThrust.value / 1000 / self.mass.value
                
                """normAcceleration = np.linalg.norm(ricAccel)
                if normAcceleration > maxAcceleration > 0:
                    ricAccel = ricAccel / np.linalg.norm(ricAccel) * maxAcceleration

                if self.deltaV_consumed[-1] <= 0.5:
                    thrustAcceleration = ricRot.T @ ricAccel  # Convert to inertial frame
                else:
                    thrustAcceleration = np.array([0.0, 0.0, 0.0])"""

                if self.OrbitHistory[-1]["RIC_Offset"][1] < 0:
                    thrustAcceleration *= -1
                
                dynamics_thrust = lambda t0, state_vector, mu: dynamics_equations(t0, state_vector, mu, thrust=thrustAcceleration)
                self.truthOrbit = self.truthOrbit.propagate(dt << u.s, method=CowellPropagator(f=dynamics_thrust))
                self.deltaV_consumed[-1] += np.linalg.norm(thrustAcceleration) * dt.value * 1000  # Convert km/s^2 to m/s^2

                rTRUTH, vTRUTH = self.truthOrbit.rv()
                ricOffset = RIC_offset([*rREF.value, *vREF.value], [*rTRUTH.value, *vTRUTH.value])[0]

                if is_within_bounds(ricOffset, self.operationBoundaries["Operation Zone"]):
                    self.OrbitStatus = "Nominal"
        
        self.OrbitHistory.append({
            "Epoch": self.epoch,
            "RIC_Offset": ricOffset,
            "OrbitStatus": self.OrbitStatus
        })
from astropy import units as u
from astropy.time import Time
from satObj import satellite
from truth2ref_orbit_animation import orbitAnimation

sat = satellite()
timeSpan = 86400 * 3  # seconds (1 day)
steps = 500    

for i in range(steps):
    sat.propagate(timeSpan / steps << u.s)

orbitHistory = sat.OrbitHistory
print(f"Total steps propagated: {len(orbitHistory)}")
print("Satellite Orbit History:")
for i in orbitHistory:
    print(i["Epoch"], i["RIC_Offset"], i["OrbitStatus"])

print("\n")
print("Delta-V Consumption:")
for i, dv in enumerate(sat.getDeltaVConsumed()):
    print(f"Step {i+1}: {dv:.6f} m/s")
print(f"Total Delta-V consumed: {sum(sat.getDeltaVConsumed()):.6f} m/s")
fig = orbitAnimation(sat.getOpsFrame(), orbitHistory)
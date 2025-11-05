from astropy import units as u

def atmDenstityData(solarCyclePhase, altitude_km):
    """Returns atmospheric density for a given solar cycle phase and altitude in km."""
    if solarCyclePhase == 'minimum':
        return densityAtSolarMinimum(altitude_km)
    elif solarCyclePhase == 'maximum':
        return densityAtSolarMaximum(altitude_km)
    elif solarCyclePhase == 'mean':
        return densityAtSolarMean(altitude_km)
    else:
        raise ValueError("Invalid solar cycle phase. Choose 'minimum', 'maximum', or 'mean'.")

def densityAtSolarMinimum(altitude_km):
    """Returns atmospheric density in kg/m^3 at solar minimum for a given altitude in km."""
    densityAtAlt = {
        100:	5.71E-07,
        123:	4.38E-08,
        145:	3.35E-09,
        168:	8.89E-10,
        190:	3.36E-10,
        213:	1.45E-10,
        235:	6.96E-11,
        258:	3.47E-11,
        280:	1.87E-11,
        303:	1.01E-11,
        325:	5.83E-12,
        348:	3.37E-12,
        370:	2.03E-12,
        393:	1.23E-12,
        415:	7.60E-13,
        438:	4.77E-13,
        460:	3.03E-13,
        483:	1.96E-13,
        505:	1.28E-13,
        528:	8.59E-14,
        550:	5.76E-14,
        573:	4.03E-14,
        595:	2.82E-14,
        618:	2.06E-14,
        640:	1.51E-14,
        663:	1.15E-14,
        685:	8.93E-15,
        708:	7.05E-15,
        730:	5.76E-15,
        753:	4.72E-15,
        775:	4.01E-15,
        798:	3.41E-15,
        820:	2.96E-15,
        843:	2.58E-15,
        865:	2.28E-15,
        888:	2.03E-15,
        910:	1.81E-15,
        933:	1.63E-15,
        955:	1.46E-15,
        978:	1.33E-15,
        1000:	1.20E-15,
    }

    if altitude_km > 1000:
        rho = 0
    elif altitude_km <= 100:
        rho = densityAtAlt[100]
    else:
        rho = min([densityAtAlt[i] for i in densityAtAlt.keys() if i < altitude_km])
    return rho # in kg/m^3

def densityAtSolarMaximum(altitude_km):
    """Returns atmospheric density in kg/m^3  at solar max for a given altitude in km."""
    densityAtAlt = {
        100:	5.70E-07,
        123:	4.50E-08,
        145:	3.56E-09,
        168:	1.03E-09,
        190:	4.28E-10,
        213:	2.06E-10,
        235:	1.10E-10,
        258:	6.14E-11,
        280:	3.65E-11,
        303:	2.19E-11,
        325:	1.38E-11,
        348:	8.76E-12,
        370:	5.75E-12,
        393:	3.79E-12,
        415:	2.54E-12,
        438:	1.72E-12,
        460:	1.18E-12,
        483:	8.18E-13,
        505:	5.70E-13,
        528:	4.02E-13,
        550:	2.84E-13,
        573:	2.05E-13,
        595:	1.47E-13,
        618:	1.08E-13,
        640:	7.89E-14,
        663:	5.86E-14,
        685:	4.39E-14,
        708:	3.32E-14,
        730:	2.56E-14,
        753:	1.97E-14,
        775:	1.56E-14,
        798:	1.24E-14,
        820:	1.01E-14,
        843:	8.23E-15,
        865:	6.85E-15,
        888:	5.76E-15,
        910:	4.91E-15,
        933:	4.24E-15,
        955:	3.68E-15,
        978:	3.24E-15,
        1000:	2.86E-15,
    }

    if altitude_km > 1000:
        rho = 0
    elif altitude_km <= 100:
        rho = densityAtAlt[100]
    else:
        rho = min([densityAtAlt[i] for i in densityAtAlt.keys() if i < altitude_km])
    return rho # in kg/m^3
def densityAtSolarMean(altitude_km):
    """Returns atmospheric density in kg/m^3  at solar mean for a given altitude in km."""

    return (densityAtSolarMaximum(altitude_km) + densityAtSolarMinimum(altitude_km)) / 2  # in kg/km^3

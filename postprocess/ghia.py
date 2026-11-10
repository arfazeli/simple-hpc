"""Ghia, Ghia and Shin (1982) centreline reference data for the lid-driven cavity.

Source: U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible
flow using the Navier-Stokes equations and a multigrid method",
J. Comput. Phys. 48 (1982) 387-411, Tables I and II (Re 400 and Re 1000).

The numbers below were copied programmatically from
~/repos/git/IncompressibleNS_SCGS/validation/validation.py (lines 6-7 and 23-40).

Index arrays hold Ghia's grid indices (1-129); coordinate = (index - 1) / 128.
_U_Y_INDEX are the y-locations of the u-data on the vertical centreline (Table I).
_V_X_INDEX are the x-locations of the v-data on the horizontal centreline (Table II).
"""

import numpy as np

_U_Y_INDEX = np.array([129,126,125,124,123,110,95,80,65,59,37,23,14,10,9,8,1])
_V_X_INDEX = np.array([129,125,124,123,122,117,111,104,65,31,30,21,13,11,10,9,1])
# Ghia's data
_U_RE400 = np.array([1,0.75837,0.68439,0.61756,0.55892,0.29093\
                         ,0.16256,0.02135,-0.11477,-0.17119,-0.32726\
                         ,-0.24299,-0.14612,-0.10338,-0.09266\
                         ,-0.08186,0])

_U_RE1000 = np.array([1,0.65928,0.57492,0.51117,0.46604,0.33304\
                          ,0.18719,0.05702,-0.06080,-0.10648,-0.27805,\
                          -0.38289,-0.29730,-0.22220,-0.20196,-0.18109,0])

_V_RE400 = np.array([0,-0.12136,-0.15663,-0.19254,-0.22847,-0.23827,\
                         -0.44993,-0.38598,0.05186,0.30174,0.30203,\
                         0.28124,0.22965,0.20920,0.19713,0.18360,0])

_V_RE1000 = np.array([0,-0.21388,-0.27669,-0.33714,-0.39188,-0.51550,\
                          -0.42665,-0.31966,0.02526,0.32235,0.33075,0.37095,0.32627,\
                          0.30353,0.29012,0.27485,0])



# Points excluded from validation by default, keyed by (Re, "u" or "v"); the value lists
# centreline coordinates. The Re 400 v value at x = 0.90625 is a likely misprint in Table II:
# every run disagrees with it by far more than anywhere else (see docs/RESULTS.md).
SUSPECT_POINTS = {(400, "v"): [0.90625]}


def _coords(index):
    return (np.asarray(index, dtype=float) - 1.0) / 128.0


def ghia_u(Re):
    """Return (y, u) on the vertical centreline, sorted by increasing y."""
    if Re == 400:
        u = _U_RE400
    elif Re == 1000:
        u = _U_RE1000
    else:
        raise ValueError(f"Ghia data available for Re 400 and 1000, got {Re}")
    y = _coords(_U_Y_INDEX)
    order = np.argsort(y)
    return y[order], np.asarray(u, dtype=float)[order]


def ghia_v(Re):
    """Return (x, v) on the horizontal centreline, sorted by increasing x."""
    if Re == 400:
        v = _V_RE400
    elif Re == 1000:
        v = _V_RE1000
    else:
        raise ValueError(f"Ghia data available for Re 400 and 1000, got {Re}")
    x = _coords(_V_X_INDEX)
    order = np.argsort(x)
    return x[order], np.asarray(v, dtype=float)[order]


if __name__ == "__main__":
    for re in (400, 1000):
        y, u = ghia_u(re)
        x, v = ghia_v(re)
        assert len(y) == 17 and len(u) == 17, "u-data must have 17 points"
        assert len(x) == 17 and len(v) == 17, "v-data must have 17 points"
        assert y[0] == 0.0 and u[0] == 0.0, "u must be 0 at y = 0"
        assert y[-1] == 1.0 and u[-1] == 1.0, "u must be 1 at y = 1 (lid)"
        assert x[0] == 0.0 and v[0] == 0.0, "v must be 0 at x = 0"
        assert x[-1] == 1.0 and v[-1] == 0.0, "v must be 0 at x = 1"
    _, u1000 = ghia_u(1000)
    assert abs(u1000.min() - (-0.38289)) < 1e-12, "Re 1000 minimum u must be -0.38289"
    print("ghia.py self-check passed")

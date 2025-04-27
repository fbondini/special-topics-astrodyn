"""Simple script for rapid first testing."""

import matplotlib.pyplot as plt
from matplotlib import style

from advanced3bodyproblem import LagrangePoints

style.use("default.mplstyle")

earth_mass = 5.9722e24  # kg
moon_mass = 7.34767309e22  # kg
earth_moon_distance = 3.844e8  # m

# Set earth-moon system lagrange points
emlp = LagrangePoints(earth_mass, moon_mass, earth_moon_distance)
print("L1 (km):", emlp.l1 / 1e3)
print("L2 (km):", emlp.l2 / 1e3)
print("L3 (km):", emlp.l3 / 1e3)
print("L4 (km):", emlp.l4 / 1e3)
print("L5 (km):", emlp.l5 / 1e3)

plt.figure(figsize=(8, 8))

plt.plot(emlp.pos1[0], emlp.pos1[1], "ok", markersize=10, label="Earth")
plt.plot(emlp.pos2[0], emlp.pos2[1], "ok", markersize=10, label="Moon")

plt.plot(emlp.l1[0], emlp.l1[1], "x", markersize=10, label="L1")
plt.plot(emlp.l2[0], emlp.l2[1], "x", markersize=10, label="L2")
plt.plot(emlp.l3[0], emlp.l3[1], "x", markersize=10, label="L3")
plt.plot(emlp.l4[0], emlp.l4[1], "x", markersize=10, label="L4")
plt.plot(emlp.l5[0], emlp.l5[1], "x", markersize=10, label="L5")

plt.title("Lagrange Points of the Earth-Moon System")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend(loc="upper left")
plt.axis("equal")  # Set equal scaling of the axis

plt.show()

# mu = emlp.mass_parameter
# x = emlp.adim_l1[0]

# print("check: ", emlp._dUdx(x, mu), "should be 0")
# print("check: ", emlp._diff_dUdx(x, mu), "should be non-zero")

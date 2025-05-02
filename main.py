"""Simple script for rapid first testing."""

import matplotlib.pyplot as plt
from matplotlib import style

from advanced3bodyproblem import ArtificialCR3BodySystem, NaturalCR3BodySystem

style.use("default.mplstyle")

earth_mass = 5.9722e24  # kg
sun_mass = 1.989e30  # kg
earth_sun_distance = 1.5063e11  # m
beta = 0.1  # solar radiation ratio

# Set earth-sun system lagrange points
em3b = NaturalCR3BodySystem(sun_mass, earth_mass, earth_sun_distance)
em3b.compute_lagrange_points()
print("Check: ", em3b.adim_l1)
print("L1 (km):", em3b.l1 / 1e3)
print("L2 (km):", em3b.l2 / 1e3)
print("L3 (km):", em3b.l3 / 1e3)
print("L4 (km):", em3b.l4 / 1e3)
print("L5 (km):", em3b.l5 / 1e3)

aem3b = ArtificialCR3BodySystem(sun_mass, earth_mass, earth_sun_distance, beta)
artificial_l2 = aem3b.adim2dim_len(aem3b.compute_generic_point(1))
print("Artificial L2 (km):", artificial_l2 / 1e3)

plt.figure(figsize=(8, 8))

plt.plot(em3b.pos1[0], em3b.pos1[1], "ok", markersize=10, label="Sun")
plt.plot(em3b.pos2[0], em3b.pos2[1], "ok", markersize=10, label="Earth")

plt.plot(em3b.l1[0], em3b.l1[1], "x", markersize=10, label="L1")
plt.plot(em3b.l2[0], em3b.l2[1], "x", markersize=10, label="L2")
plt.plot(em3b.l3[0], em3b.l3[1], "x", markersize=10, label="L3")
plt.plot(em3b.l4[0], em3b.l4[1], "x", markersize=10, label="L4")
plt.plot(em3b.l5[0], em3b.l5[1], "x", markersize=10, label="L5")

plt.plot(
    artificial_l2[0], artificial_l2[1], "x", markersize=10, label="Artificial L2",
)

plt.title("Natural and Artificial Lagrange Points of the Sun-Earth System")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend(loc="upper left")
plt.axis("equal")  # Set equal scaling of the axis

plt.show()

# mu = em3b.mass_parameter
# x = em3b.adim_l1[0]

# print("check: ", em3b._dUdx(x, mu), "should be 0")
# print("check: ", em3b._diff_dUdx(x, mu), "should be non-zero")

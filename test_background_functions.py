from functions import (
    PolynomialBackgroundFunction,
    ExponentialBackgroundFunction,
    SaturationBackgroundFunction,
)
import numpy as np
import matplotlib.pyplot as plt
import os

print("== Test Background Funtions ==\n")
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
[ax.set_ylabel("intensity / counts") for ax in axs[:, 0]]
[ax.set_xlabel("radial cord / pixel") for ax in axs[-1, :]]
dummy_x = np.arange(1500)  #  pixel along radial axis

# linear background function
print("-> testing LinearBackgroundFunction")
ax = axs.ravel()[0]
ax.set_title("Linear")
dummy_y = 50 + (0.1 * dummy_x)
dummy_y += dummy_y * np.random.normal(0, 0.1, dummy_x.size)
ax.plot(dummy_x, dummy_y)
lin_bg = PolynomialBackgroundFunction(1)
lin_bg.fit(dummy_x, dummy_y)
print(lin_bg)
ax.plot(dummy_x, lin_bg.eval(dummy_x))
print("-" * 31, end="\n")


# quadratic background function
print("-> testing QuadraticBackgroundFunction")
ax = axs.ravel()[1]
ax.set_title("Quadratic")
dummy_y = 50 - 0.03 * dummy_x + 0.0001 * dummy_x**2
dummy_y += dummy_y * np.random.normal(0, 0.1, dummy_x.size)
ax.plot(dummy_x, dummy_y)
quad_bg = PolynomialBackgroundFunction(2)
quad_bg.fit(dummy_x, dummy_y)
print(quad_bg)
ax.plot(dummy_x, quad_bg.eval(dummy_x))
print("-" * 31, end="\n")

# exponential background function
print("-> testing ExponentialBackgroundFunction")
ax = axs.ravel()[2]
ax.set_title("Exponential")
dummy_y = 50 + np.exp(0.00352 * dummy_x)
dummy_y += dummy_y * np.random.normal(0, 0.1, dummy_x.size)
ax.plot(dummy_x, dummy_y)
exp_bg = ExponentialBackgroundFunction()
exp_bg.fit(dummy_x, dummy_y)
print(exp_bg)
ax.plot(dummy_x, exp_bg.eval(dummy_x))
print("-" * 31, end="\n")

# saturation background function
print("-> testing SaturationBackgroundFunction")
ax = axs.ravel()[3]
ax.set_title("Saturation")
dummy_y = 50 + 2 * (dummy_x**0.6)
dummy_y += dummy_y * np.random.normal(0, 0.1, dummy_x.size)
ax.plot(dummy_x, dummy_y)
sat_bg = SaturationBackgroundFunction()
sat_bg.fit(dummy_x, dummy_y)
print(sat_bg)
ax.plot(dummy_x, sat_bg.eval(dummy_x))
print("-" * 31, end="\n")

# plt.savefig(f".{os.sep}figures.readme{os.sep}background_functions.png", dpi=200)
plt.show()

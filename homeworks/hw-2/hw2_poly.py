#############################################
#   By: Ahmed Abu Qahf                      #
#   Date: 4/2/2023                          #
#   ASU ID# 1223023698                      #
#   Prof. Dr. Steve Millman                 #
#############################################
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def func(x, a, b, c):
    return a * x**2 + b * x + c

STEP = 0.01
START_R = 0
END_R = 5 + STEP

r_vals = np.arange(START_R, END_R, STEP)
integrals = np.zeros(len(r_vals), float)

# first integration with a = 2, b = 3, c = 4
for val in range(len(r_vals)):
    integrals[val], err = quad(func, 0, r_vals[val], args=(2, 3, 4))

# plot first integration results
plt.plot(r_vals, integrals, label="a = 2, b = 3, c = 4")

# second integration with a = 2, b = 1, c = 1
for val in range(len(r_vals)):
    integrals[val], err = quad(func, 0, r_vals[val], args=(2, 1, 1))

plt.plot(r_vals, integrals, label="a = 2, b = 1, c = 1")
plt.xlabel("Value of 'r'")
plt.ylabel("ax^2 + bx + c")
plt.title("hw2 poly")
plt.legend()
plt.show()
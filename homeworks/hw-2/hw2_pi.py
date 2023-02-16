#############################################
#   By: Ahmed Abu Qahf                      #
#   Date: 4/2/2023                          #
#   ASU ID# 1223023698                      #
#   Prof. Dr. Steve Millman                 #
#############################################

# after substituting x = z / 1 - z
# the integral have become 1 / sqrt(1-z) * sqrt(z) from 0 to 1 after simplification
import numpy as np
from scipy.integrate import quad

def func(z):
    return 1 / (np.sqrt(1 - z) * np.sqrt(z))

integral, error = quad(func, 0, 1)

print("Pi is {:,.8f}".format(integral))
print("Difference from numpy.pi is: {:,.15f}".format(np.pi - integral))
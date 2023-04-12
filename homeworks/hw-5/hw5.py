import numpy as np                  # for arrays
import matplotlib.pyplot as plt     # for plotting
from scipy.integrate import odeint  # for solving ordinary differential equations

##################################################################################
# plot_solution:: plots y vs t                                                   #
##################################################################################
def plot_solution(time, sol, title):
    plt.plot(time, sol)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

##################################################################################
# time_range:: returns a range of 700 points between 0 & 7                       #
##################################################################################
def time_range():
    return np.linspace(0, 7, 700)

##################################################################################
# ode_first:: y` = cos(t)                                                        #
##################################################################################
def ode_first(y, t):
    return np.cos(t)

##################################################################################
# ode_first:: y` = cos(t)                                                        #
##################################################################################
def problem1_sol():
    y_init = 1
    time = time_range()
    solution = odeint(ode_first, y_init, time)
    print(f"Solution of y'=cos(t) with y(0)=1 is: {solution}")
    plot_solution(time, solution, "Solution of y'=cos(t) with y(0)=1")

##################################################################################
# main:: project entry point                                                     #
##################################################################################
def main():
    # solve 1st problem
    problem1_sol()

# call main
main()

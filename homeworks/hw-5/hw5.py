import numpy as np                  # for arrays
import matplotlib.pyplot as plt     # for plotting
from scipy.integrate import odeint  # for solving ordinary differential equations

##################################################################################
# plot_solution:: plots y vs t                                                   #
##################################################################################
def plot_solution(time, sol, title):
    plt.plot(time, sol)
    plt.xlabel('t values')
    plt.ylabel('y values')
    plt.title(title)
    plt.show()

##################################################################################
# time_range:: returns a range of 700 points between 0 & 7                       #
##################################################################################
def time_range():
    return np.linspace(0, 7, 700)

##################################################################################
# ode1:: y' = cos(t)                                                        #
##################################################################################
def ode1(y, t):
    return np.cos(t)

##################################################################################
# ode2:: y' = -y+t^2e^-2t+10                                                     #
##################################################################################
def ode2(y, t):
    return -y + t**2 * np.exp(-2*t) + 10

##################################################################################
# problem1_sol:: first problem solver                                            #
##################################################################################
def problem1_sol():
    y_init = 1
    time = time_range()
    solution = odeint(ode1, y_init, time)
    plot_solution(time, solution, "Solution of y'=cos(t) with y(0)=1")

##################################################################################
# problem2_sol:: second problem solver                                           #
##################################################################################
def problem2_sol():
    y_init = 0
    time = time_range()
    solution = odeint(ode2, y_init, time)
    plot_solution(time, solution, "Solution of y'=-y+t^2e^-2t+10 with y(0)=0")

##################################################################################
# main:: project entry point                                                     #
##################################################################################
def main():
    # solve 1st problem
    problem1_sol()

    # solve 2nd problem
    problem2_sol()

# call main
main()

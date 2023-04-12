import numpy as np                  # for cos, sin & exp
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
# ode3:: y" + 4y' + 4y = 25cos(t) + 25sin(t)                                     #
##################################################################################
def ode3(y, t):
    dy = y[1]   # y'
    dyy = 25 * np.cos(t) + 25 * np.sin(t) - 4 * y[1] - 4 * y[0] # y"
    return [dy, dyy]

##################################################################################
# problem1_sol:: first problem solver                                            #
##################################################################################
def problem1_sol():
    y_init = 1              # initial value
    time = time_range()     # 700 points from 0-7

    # solve by odeint
    solution = odeint(ode1, y_init, time)

    # plot solution
    plot_solution(time, solution, "Solution of y'=cos(t) with y(0)=1")

##################################################################################
# problem2_sol:: second problem solver                                           #
##################################################################################
def problem2_sol():
    y_init = 0              # initial value
    time = time_range()     # 700 points from 0-7 

    # solve by odeint
    solution = odeint(ode2, y_init, time)

    # plot solution
    plot_solution(time, solution, "Solution of y'=-y+t^2e^-2t+10 with y(0)=0")

##################################################################################
# problem3_sol:: second problem solver                                           #
##################################################################################
def problem3_sol():
    y_init = [1, 1]         # initial value
    time = time_range()     # 700 points from 0-7

    # solve using odeint
    solution = odeint(ode3, y_init, time)

    # extract value for y & y prime
    y = solution[:, 0]
    y_prime = solution[:, 1]

    # plot y vs t & y prime vs t as well
    plt.plot(time, y, label='y')
    plt.plot(time, y_prime, label='y prime')
    plt.xlabel('t values')
    plt.ylabel('y values')
    plt.title("Solution of y\"+4y'+4y=25cos(t)+25sin(t) with y(0)=1, y'(0)=1")
    plt.legend()
    plt.show()

##################################################################################
# main:: project entry point                                                     #
##################################################################################
def main():
    # solve 1st problem
    problem1_sol()

    # solve 2nd problem
    problem2_sol()

    # solve 3rd problem
    problem3_sol()

# call main
main()

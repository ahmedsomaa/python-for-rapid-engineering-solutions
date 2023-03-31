from scipy.optimize import fsolve, leastsq  # to find voltage across diode
import matplotlib.pyplot as plt             # for plotting
import numpy as np                          # for log10, exp, arange, loadtxt
import warnings                             # to ignore warnings

# global constants
K = 1.380648e-23        # boltzman's constant
Q = 1.6021766208e-19    # coulomb's constant
MAX_TOLERANCE = 1e-4    # maximum allowed tolerance
MAX_ITERATIONS = 100    # maximum allowed iterations
SRC_V_MIN = 0.1         # diode 1 applied voltage start
SRC_V_MAX = 2.5         # diode 1 applied voltage end
NORM_CONST = 1e-15      # constant used in normalized error calculation

# problem 1 parameters
N1 = 1.7                # ideality
T1 = 350                # temperature
R1 = 11000              # resistor
IS1 = 1e-9              # source current

# problem 2 parameters
A = 1e-8                # sectional area
T2 = 375                # temperature

# unknown parameters initial guess
opt_n = 1.5             # optimal ideality initial guess
opt_r = 10000           # optimal resistor optimal guess
opt_phi = 0.8           # optimal barrier height initial guess
vd_init = 0.1           # diode voltage initial guess

##################################################################################
# print_header:: print problem name header to the screen                         #
##################################################################################
def print_header(prob):
    print('-' * 18)
    print(f'Problem {prob} Solution')
    print('-' * 18)

##################################################################################
# print_line:: print +{"-"*12}+{"-"*21}+{"-"*21}+{"-"*21}+{"-"*25}+ to screen    #
##################################################################################
def print_line():
    print(f'+{"-"*12}+{"-"*21}+{"-"*21}+{"-"*21}+{"-"*25}+')

##################################################################################
# diode_1:: equation of current passing through the first diode                  #
##################################################################################
def diode_1(Vd):
    return IS1 * (np.exp((Vd * Q) / (N1 * K * T1)) - 1)

##################################################################################
# diode1_nodal:: equation of nodal analysis                                      #
##################################################################################
def diode1_nodal(Vd, Vs):
    return ((Vd - Vs) / R1) + diode_1(Vd)

##################################################################################
# diode1_plot:: plots log diode current vs source voltage & diode voltage        #
##################################################################################
def diode1_plot(log_diode_curr, source_volt, diode_volt):
    plt.title('Problem 1 Plot')
    plt.plot(source_volt, log_diode_curr,
             label='log(Diode Current) vs Source Voltage')
    plt.plot(diode_volt, log_diode_curr,
             label='log(Diode Current) vs Diode Voltage')
    plt.xlabel('Voltage (volt)')
    plt.ylabel('Diode current (log scale)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

##################################################################################
# diode2_nodal:: equation of nodal analysis                                      #
##################################################################################
def diode2_nodal(diode_v, src_v, r_value, ide_value, temp, src_i):
    # diode current equation constant
    vt = (ide_value * K * temp) / Q

    # diode current equation
    diode_curr = src_i * (np.exp(diode_v / vt) - 1)

    # nodal function
    return ((diode_v - src_v) / r_value) + diode_curr

##################################################################################
# solve_diode2_current:: calculate diode2 current using initial guesses          #
##################################################################################
def solve_diode2_current(area, phi_value, r_value, ide_value, temp, src_v):
    diode_volt_est = np.zeros_like(src_v)
    diode_curr = np.zeros_like(src_v)

    # specify initial diode voltage for fsolve
    vd_guess = vd_init
    sat_i = area * temp * temp * np.exp((-phi_value * Q / (K * temp)))

    # calculate diode voltage for every given voltage by solving nodal analysis
    for i in range(len(src_v)):
        vd_guess = fsolve(diode2_nodal, vd_guess,
                          args=(src_v[i], r_value, ide_value, temp, sat_i), xtol=1e-12)[0]

        # append it to diode volt array
        diode_volt_est[i] = vd_guess

    # compute diode current
    vt = (ide_value * K * temp) / Q
    diode_curr = sat_i * (np.exp(diode_volt_est / vt) - 1)
    return diode_curr

##################################################################################
# optimize_r:: optimization for the resistor                                     #
##################################################################################
def optimize_r(r_value, phi_value, ide_value, area, temp, src_v, meas_i):
    # compute diode current using optimized params
    diode_curr = solve_diode2_current(
        area, phi_value, r_value, ide_value, temp, src_v)

    # return absolute error
    return diode_curr - meas_i

##################################################################################
# optimize_phi:: optimization for the barrier height                             #
##################################################################################
def optimize_phi(phi_value, r_value, ide_value, area, temp, src_v, meas_i):
    # compute diode current using optimized params
    diode_curr = solve_diode2_current(
        area, phi_value, r_value, ide_value, temp, src_v)

    # return absolute error
    return (diode_curr - meas_i) / (diode_curr + meas_i + NORM_CONST)

##################################################################################
# optimize_n:: optimization for the ideality                                     #
##################################################################################
def optimize_n(ide_value, r_value, phi_value, area, temp, src_v, meas_i):
    # compute diode current using optimized params
    diode_curr = solve_diode2_current(
        area, phi_value, r_value, ide_value, temp, src_v)

    # return absolute error
    return (diode_curr - meas_i) / (diode_curr + meas_i + NORM_CONST)

##################################################################################
# diode2_plot:: plots log diode current vs source voltage & diode voltage        #
##################################################################################
def diode2_plot(src_v, meas_i, pred_i):
    fig, ax = plt.subplots()

    ax.set_title('Problem 2 Plot')
    ax.set_xlabel('Voltage (volts)')
    ax.set_ylabel('Measure Current (log scale)', c='tab:blue')
    ax.plot(src_v, np.log10(meas_i), marker='o', markerfacecolor='tab:blue',
            c='tab:blue', label='log(Diode Current) vs Source Voltage')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    plt.grid()

    ax2 = ax.twinx()
    ax2.set_ylabel('Predicted Current (log scale)', c='tab:orange')
    ax2.plot(src_v, np.log10(pred_i), marker='+', markerfacecolor='tab:orange',
             c='tab:orange', label="log(Predicted Current) cs Source Voltage")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    fig.legend()
    plt.grid()
    plt.show()


# --------------------------------------------------------------- Project Starts Here
# ignore the warnings generated by leastsq operations
warnings.simplefilter("ignore")

# --------------------------------------------------------------- Problem 1
print_header(1)
diode_volt = []     # store diode voltage
diode_curr = []     # store diode current

# create a src_v array from 0.1 -> 2.5 with step size of 0.1
src_v = np.arange(SRC_V_MIN, SRC_V_MAX + SRC_V_MIN, SRC_V_MIN, dtype=float)

for v in src_v:
    # compute diode voltage for each step
    vd_init = fsolve(diode1_nodal, vd_init, args=(v,))[0]
    # add it to diode voltage list
    diode_volt.append(vd_init)

    # compute diode current for calculated Vd
    diode_i = diode_1(vd_init)
    # append it to diode current list
    diode_curr.append(diode_i)

print('diode voltage:')
print(diode_volt)
print('\ndiode current:')
print(diode_curr)

# draw the required plot
diode1_plot(np.log10(diode_curr), src_v, diode_volt)

print()
# --------------------------------------------------------------- Problem 2
# solve problem 2
print_header(2)

# load DiodeIV.txt file
data = np.loadtxt('DiodeIV.txt', dtype=np.float64)
src_volt = np.array([row[0] for row in data])       # store diode voltage
curr_meas = np.array([row[1] for row in data])      # store diode current

# keep track of iterations before convergence
iters = 0

# calculate diode current using initial guess
curr_pred = solve_diode2_current(A, opt_phi, opt_r, opt_n, T2, src_volt)

# compute normalized error
norm_err = np.linalg.norm((curr_pred - curr_meas) /
                          (curr_pred + curr_meas + NORM_CONST), ord=1)

# prepare table header for iterations, opt_r, opt_phi, opt_n & normalized error
print('\nIterations & residual errors table')
print_line()
print('| Iterations |          R          |         Phi         |          n          |     Residual Error      |')
print_line()

while (norm_err > MAX_TOLERANCE and iters < MAX_ITERATIONS):
    # updater iterations
    iters += 1

    # optimize resistance
    opt_r = leastsq(optimize_r, opt_r, args=(
        opt_phi, opt_n, A, T2, src_volt, curr_meas))[0][0]

    # optimize barrier height
    opt_phi = leastsq(optimize_phi, opt_phi, args=(
        opt_r, opt_n, A, T2, src_volt, curr_meas))[0][0]

    # optimize ideality
    opt_n = leastsq(optimize_n, opt_n, args=(
        opt_r, opt_phi, A, T2, src_volt, curr_meas))[0][0]

    # compute diode current
    curr_pred = solve_diode2_current(A, opt_phi, opt_r, opt_n, T2, src_volt)

    # compute normalized error
    norm_err = np.linalg.norm(
        (curr_pred - curr_meas) / (curr_pred + curr_meas + NORM_CONST), ord=1)

    # print iterations, opt_r, opt_phi, opt_n & normalized error
    print(f'| {iters: <{10}} | {opt_r: <{19}} | {opt_phi: <{19}} | {opt_n: <{19}} | {norm_err: <{23}} |')
    print_line()

# draw the required plot
diode2_plot(src_volt, curr_meas, curr_pred)

# print optimized values
print('\nOptimized parameters table')
print('+-----------+-----------------+')
print('| Parameter | Optimized Value |')
print('+-----------+-----------------+')
print(f'| R         | {round(opt_r, 4): <{15}} |')
print('+-----------+-----------------+')
print(f'| Phi       | {round(opt_phi, 4): <{15}} |')
print('+-----------+-----------------+')
print(f'| n         | {round(opt_n, 4): <{15}} |')
print('+-----------+-----------------+')

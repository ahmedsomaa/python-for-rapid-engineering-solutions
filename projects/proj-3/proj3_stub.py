################################################################################
# This function does the optimization for the resistor                         #
# Inputs:                                                                      #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################

def opt_r(r_value,ide_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return meas_i - diode_i


################################################################################
# This is how leastsq calls opt_r                                              #
################################################################################

    r_val_opt = optimize.leastsq(opt_r,r_val,
                                 args=(ide_val,phi_val,P2_AREA,P2_T,
                                       source_v,meas_diode_i))
    r_val = r_val_opt[0][0]


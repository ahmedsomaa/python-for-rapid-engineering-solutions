#############################################################
#   By: Ahmed Abu Qahf                                      #
#   Data: 2/14/2023                                         #
#   ASU ID# 1223023698                                      #
#   Class: EEE 591 - Python for Rapid Engineering Solutions #
#   Instructor: Steve Millman                               #
#   File: hw4.py                                            #
#############################################################

import numpy as np                      # get numpy
import PySimpleGUI as sg                # get the higher-level GUI package
import matplotlib.pyplot as plt         # get matplotlib for plotting

MAX_YEARS = 70                          # years to work and have wealth
ANALYSIS_NO = 10                        # number of times to run the analysis

# define the field names and their indices
FN_MEAN_RETURN = 'Mean Return (%)'
FN_STANDARD_DEVIATION = 'Std Dev Return (%)'
FN_YEARLY_CONTRIBUTION = 'Yearly Contribution ($)'
FN_CONTRIBUTION_YEARS = 'No. of Years of Contribution'
FN_RETIREMENT_YEARS = 'No. of Years to Retirement'
FN_ANNUAL_SPEND_IN_RETIREMENT = 'Annual Spend in Retirement'
FN_WEALTH_AFTER_RETIREMENT = 'Retirement Wealth'

# fields list
FIELD_NAMES = [FN_MEAN_RETURN, FN_STANDARD_DEVIATION, FN_YEARLY_CONTRIBUTION, FN_CONTRIBUTION_YEARS, 
FN_RETIREMENT_YEARS, FN_ANNUAL_SPEND_IN_RETIREMENT]

NUM_FIELDS = 6                      # how many fields there are

# define buttons
B_QUIT = 'Quit'                     # quit button
B_CALCULATE = 'Calculate'           # calculate button

# define the matrix to hold wealth per analysis
wealth_matrix = np.zeros((MAX_YEARS, ANALYSIS_NO), dtype=float)

###############################################################################################################
# Function to compute average wealth, plot wealth as a function of year for each of the 10 analyses           #
# Inputs:                                                                                                     #
#    window  - the top-level widget                                                                           #
#    entries - the dictionary of entry fields                                                                 #
# Output:                                                                                                     #
#    only output is to screen (for debug) and the GUI and the avg of our wealth at retirement                 #
###############################################################################################################
def wealth_calculator(window, entries):
    # retrieve value for each text input and convert to float
    r = float(entries[FN_MEAN_RETURN])                                      # mean return rate 
    Y = float(entries[FN_YEARLY_CONTRIBUTION])                              # yearly contribution
    S = float(entries[FN_ANNUAL_SPEND_IN_RETIREMENT])                       # annual spend after retirement
    sigma = float(entries[FN_STANDARD_DEVIATION])                           # std dev of investment annual volatility 
    nyears_retirement = float(entries[FN_RETIREMENT_YEARS])                 # years to retirement
    nyears_contributions = float(entries[FN_CONTRIBUTION_YEARS])            # contribution years

    # plot in single figure
    plt.figure()
    for n in range(ANALYSIS_NO):
        last = 0                                                            # assume last index is 0
        current = 0                                                         # startwith $0 for each calculation
        noise = (sigma / 100) * np.random.randn(MAX_YEARS)                  # generate random noise
        for i in range(MAX_YEARS):
            # check if current year is less than year of contribution
            # from start until contributions end
            if i < nyears_contributions:
                current = current * (1 + (r/ 100) + noise[i]) + Y
            # from end of contribution till retirement
            elif i >= nyears_contributions and i < nyears_retirement:
                current = current * (1 + (r/ 100) + noise[i])
            # from retirement to end
            elif i >= nyears_retirement:
                current = current * (1 + (r/ 100) + noise[i]) - S
            
            # if the wealth for current year < 0, stop analysis
            if (current >= 0):
                wealth_matrix[i, n] = current
                last = i
            else:
                # money ran out
                break
        # plot data points for on the analysis line
        line, = plt.plot(range(last + 1), wealth_matrix[0: last + 1, n], '-x')
        line.set_label('Analysis ' + str(n + 1))
        plt.title("Wealth Over 70 Years")
        plt.xlabel("years")
        plt.ylabel("wealth")
        plt.grid(True)

    # calculate the average for the 10 analysis, each one for 70 years
    # first calculae the total at the retirement year
    # withdrawals start the year after retirement
    retirement_total = np.sum(wealth_matrix[int(nyears_retirement) + 1, :], axis=0)
    wealth_average = retirement_total / ANALYSIS_NO
    
    # display the result to screen
    window[FN_WEALTH_AFTER_RETIREMENT].Update("Wealth at retirement: ${:,.2f}".format(wealth_average))
    
    # show plot here to dispaly all graphs with legend
    plt.legend()
    plt.show(block=False)

##################################################################### GUI layout
# set the window theme
sg.theme('DefaultNoMoreNagging')

# set the font and size
sg.set_options(font=('Helvetica', 20))

# start with empty list
layout = []

# for each field, append a label and text input on the same row
for index in range(NUM_FIELDS):
    layout.append([sg.Text(FIELD_NAMES[index], size=(25, 1)), sg.InputText(key=FIELD_NAMES[index], size=(20, 1))])

# append initial text for the wealth after retirement
layout.append([sg.Text('', key=FN_WEALTH_AFTER_RETIREMENT, size=(30, 1))])

# append quit & calculate buttons
layout.append([sg.Button(B_QUIT), sg.Button(B_CALCULATE)])

# start the window manager
window = sg.Window('Retirement Wealth Calculator', layout)

# run the event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == B_QUIT:
        break
    if event == B_CALCULATE:
        wealth_calculator(window, values)

window.close()
################################################################################
# Created on Mon Feb 06                                                        #
#                                                                              #
# @author: aabuqahf@asu.edu;                                                   #
#                                                                              #
# Program to solve resister network with voltage and/or current sources        #
################################################################################
import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]

################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################

def get_dimensions(netlist):           # pass in the netlist
    node_cnt = 0                       # initialize nodes count with 0
    volt_cnt = 0                       # initialize voltage count with 0
    for comp in netlist:
        # count the number of nodes
        # the count will always be the greatest number of i & j
        # since we're ignoring the ground node (0)
        if (comp[COMP.I] != comp[COMP.J]):
            node_cnt = max(comp[COMP.I], comp[COMP.J], node_cnt)
        else:
            print('Wrong node values, cannot move from node', comp[COMP.I], 'to node', comp[COMP.J])
        if comp[COMP.TYPE] == COMP.VS:          # count number of voltage sources
            volt_cnt += 1
    # print('Nodes:', node_cnt, 'Voltage sources:', volt_cnt)
    return node_cnt, volt_cnt

################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
################################################################################

def stamper(y_add, netlist, currents, node_cnt):
    # return the total number of rows in the matrix for
    # error checking purposes
    # add 1 for each voltage source...

    for comp in netlist:                  # for each component...
        #print(' comp ', comp)            # which one are we handling...

        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1

        if (comp[COMP.TYPE] == COMP.R):           # if component is a resistor
            if (i >= 0):                            # add 1/VAL for entries [i,i] & [j,j]
                y_add[i, i] += 1.0 / comp[COMP.VAL]
            if (j >= 0):                            # add 1/VAL for entries [i,i] & [j,j]
                y_add[j, j] += 1.0 / comp[COMP.VAL]
            if (i >= 0 and j >= 0):                 # subtract 1/VAL for entries [i,j] & [j,i]
                y_add[i, j] -= 1.0 / comp[COMP.VAL]
                y_add[j, i] -= 1.0 / comp[COMP.VAL]
        
        elif (comp[COMP.TYPE] == COMP.VS):          # if component is a voltage source
            node_cnt += 1
            M = node_cnt                            # Number of rows in admittance matrix
            if (i>= 0):                             # Set entries [M,i] and [i,M] to 1
                y_add[M-1, i] = 1.0
                y_add[i, M-1] = 1.0
            if (j >= 0):                            # Set entries [M,j] and [j,M] to -1
                y_add[M-1, j] = -1.0
                y_add[j, M-1] = -1.0

            # Add another row to the currents matrix
            # Set entry [M] to VAL
            currents[M-1] = comp[COMP.VAL]

            # Add another row to the voltage matrix
            # Set entry [M] to 0
            voltages[M-1] = 0

        elif (comp[COMP.TYPE] == COMP.IS):          # if component is a current source
            if (i >= 0):                            # Subtract I at entry [i] in the current matrix
                if (comp[COMP.VAL] >= 0):
                    currents[i] -= 1.0 * comp[COMP.VAL]
                else:
                    currents[i] += 1.0 * comp[COMP.VAL]
            if (j >= 0):                            # Add I at entry [j] in the current matrix
                if(comp[COMP.VAL] >= 0):
                    currents[j] += 1.0 * comp[COMP.VAL]
                else:
                    currents[j] -= 1.0 * comp[COMP.VAL]
    
    # print('Admittance Matrix =', y_add)
    # print('Voltage Matrix =', voltages)
    # print('Currents Matrix =', currents)
    return node_cnt  # should be same as number of rows!

################################################################################
# Start the main program now...                                                #
################################################################################

# Read the netlist!
netlist = read_netlist()

# Print the netlist so we can verify we've read it correctly
# for index in range(len(netlist)):
#     print(netlist[index])
# print("\n")

# Get the count of nodes and voltage sources and print them
nodes_count, voltage_sources_count = get_dimensions(netlist)

# Get the total number of nodes
total = nodes_count + voltage_sources_count

# Prepare admittance, currents, voltage lists
admittance = np.zeros((total, total), dtype=float)
currents = np.zeros(total, dtype=float)
voltages = np.zeros(total, dtype=float)

# Call stamper function to return number of nodes
node_count = stamper(admittance, netlist, currents, nodes_count)
# print("Nodes count from stamper function =", node_count)

# Solve for voltages
voltages = solve(admittance, currents)
print(voltages)
# print('Voltage Vector =', voltages)
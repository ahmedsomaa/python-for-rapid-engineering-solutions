################################################################################
# Created on Mon Jan 21, 2019                                                  #
#                                                                              #
# @author: sdm                                                                 #
#                                                                              #
# Constants for solving resistor networks with voltage and/or current sources  #
################################################################################

# Define some constants we'll use to reference things
RESIS = 'R'      # a resistor
V_SRC = 'VS'     # a voltage source
I_SRC = 'IS'     # a current source

# Define the list data structure that we'll use to hold components:
# [ Type, Name, i, j, Value ] ; set up an index for each component's property
TYPE = 0         # voltage source or resister
NAME = 1         # name of the component
I    = 2         # "from" node of the component 
J    = 3         # "to" node of the component
VAL  = 4         # value of the component

# Define the different types of component
R    = 0         # A resistor
VS   = 1         # An independent voltage source
IS   = 2         # An independent current source

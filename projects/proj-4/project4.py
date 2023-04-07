import numpy as np      # package needed to read the results file
import subprocess       # package needed to lauch hspice
import os               # package needed to make sure file exists


MAX_FAN = 8     # max value for fan 
MAX_INV = 12    # max value for number of ineverters

INPUT_FILE_NAME = 'InvChainTemp.sp'     # input filename
OUTPUT_FILE_NAME = 'InvChain.mt0.csv'   # output filename
HSPICE_INPUT_FILE_NAME = 'InvChain.sp'  # file to use in hspice simulation

################################################################################
# extract_tphl: return tphl value from hspice out file                         #
################################################################################
def extract_tphl():
    data = np.recfromcsv(OUTPUT_FILE_NAME, comments="$", skip_header=3)
    tphl = data["tphl_inv"]
    return tphl

################################################################################
# run_hspice: run hspice simulation for fan & no of inverters                  #
################################################################################
def run_hspice():
    # launch hspice. Note that both stdout and stderr are captured so
    # they do NOT go to the terminal!
    proc = subprocess.Popen(["hspice", HSPICE_INPUT_FILE_NAME],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output, err = proc.communicate()

################################################################################
# prepare_simulation_input: generate netlist for fan & inverters combination   #
################################################################################
def prepare_simulation_input(netlist, fan, N):
    # add .param line to hspice file
    netlist += f'\n\n.param fan = {fan}\n'

    # add inverters
    # if one inverter, just add from a -> z
    if (N == 1):
        netlist += 'Xinv1 a z inv M=1\n'
    else:
        # start ascii number for node a
        start = ord('a')
        # add first inverter a -> b
        netlist += 'Xinv1 a b inv M=1\n'
        for inv in range(2, N):
            # add next inverter
            netlist += f'Xinv{inv} {chr(start+1)} {chr(start+2)} inv M=fan**{inv-1}\n'
            # increase current node char ASCII code
            start += 1
        # add last inverter
        netlist += f'Xinv{N} {chr(start+1)} z inv M=fan**{N}\n'
    # end netlist
    netlist += '.end'

    # write netlist to file
    file = open(HSPICE_INPUT_FILE_NAME, 'w')
    file.write(netlist)

################################################################################
# read_netlist: read hspice initial file                                       #
################################################################################
def read_netlist():
    file = open(INPUT_FILE_NAME, 'r')
    return file.read()

##################################################################################
# main:: project entry point                                                     #
##################################################################################
def main():
    # create fan list & num of inverters list
    fan_list = [fan for fan in range(2, MAX_FAN)]                   # fan list
    inv_list = [inv for inv in range(0, MAX_INV) if inv % 2 != 0]   # inv list

    # prepare minimum delay & optimal fan & num of inverters
    min_delay = float('inf')    # minimum delay initial value
    optimal_fan = 0             # initial value for optimal fan
    optimal_N = 0               # optimal value for optimal num of inverters

    # read netlist
    netlist = read_netlist()

    # loop over fan & inverters combinations
    print('+-----+-----+------------+')
    print('| Fan | N   | tphl       |')
    print('+-----+-----+------------+')
    for fan in fan_list:
        for inv in inv_list:
            # prepare netlist hspice file for each fan & #of inverters
            prepare_simulation_input(netlist, fan, inv)

            # run hspace simulation
            run_hspice()

            # make sure output file is ready
            output_ready = False
            while(not output_ready):
                output_ready = os.path.isfile(OUTPUT_FILE_NAME)

            # output file is ready, extract tphl
            tphl = extract_tphl()

            print(f'| {fan: <{3}} | {inv: <{3}} | {tphl: <{10}} |')
            print('+-----+-----+------------+')

            if (tphl < min_delay):
                min_delay = tphl
                optimal_fan = fan
                optimal_N = inv
    
    print('\nBest values were:')
    print(f'fan = {optimal_fan}')
    print(f'num_inverters = {optimal_N}')
    print(f'tphl = {min_delay}')

# start app
main()

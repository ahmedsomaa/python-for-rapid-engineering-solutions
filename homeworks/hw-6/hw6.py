import random           # to generate random points
import numpy as np      # to compute the distance of each point from the origin

#------------------------------------------------ global constants
TRIALS = 100                                    # Number of trials per precision
PRECISIONS = [10**(-i) for i in range(1, 8)]    # List of precisions to try
MAXIMUM_POINTS = 10000                          # Maximum number of points to use

################################################################################
# generate_r:: generates random value for r                                    #
################################################################################
def generate_r():
    # generate a random point
    x = random.uniform(0, 1)    # random value for x between 0 & 1
    y = random.uniform(0, 1)    # random value for y between 0 & 1

    # calculate the value of r
    return np.sqrt(x**2 + y**2)

################################################################################
# print_results:: prints average pi & num of attempts for each precision       #
################################################################################
def print_results(attemps, pi_val, precision):
    if attemps > 0:
        pi_avg = pi_val / attemps
        print(f'{precision} success {attemps} times {pi_avg}')
    else:
        print(f'{precision} no success')

################################################################################
# main:: project entry point                                                   #
################################################################################
def main():
    for precision in PRECISIONS:
        success_attemps = 0     # init number of successfull attempts
        pi_val = 0              # init the value of pi

        for _ in range(TRIALS):
            trials = 0      # init number of trials
            inside = 0   # init number of total points inside

            while trials < MAXIMUM_POINTS:
                # generate random value for r
                r = generate_r()

                # check if r is inside the circle or not
                if (r <= 1):
                    inside += 1      # increment number of points inside

                trials += 1

                # check if we achieved the precision
                if abs(4*inside/trials - np.pi) < precision:
                    success_attemps += 1
                    pi_val += 4*inside/trials
                    break

        # print results
        print_results(success_attemps, pi_val, precision)

# start script
main()
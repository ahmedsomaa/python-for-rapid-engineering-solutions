################################################################################
# Function prime_numbers_generator generates prime numbers within a given range#
# inputs:                                                                      #
# start: the first number in the range                                         #
# end: the last number in the range                                            #
# outputs:                                                                     #
# prime_numbers_list: list of the primes found within the given range          #
################################################################################
import numpy as np

def prime_numbers_generator(start, end):
    prime_numbers_list = [2]                        # initialize prime list with 2
    # loop over the numbers to check for prime numbers
    # additional one is added to include end within the range
    for num in range(start, end + 1):
        for prime in prime_numbers_list:            # loop over primes to check if current number is divisible by prime
            if num % prime == 0:                    # if current number is divisible by any prime in the list, then it
                break                               # is not a prime number and shouldn't be added to the primes' list

            elif prime > np.sqrt(num):              # if the current prime is greater than sqrt(num), then it is a prime
                prime_numbers_list.append(num)      # and should be added to the prime's list
                break

    return prime_numbers_list

print(prime_numbers_generator(3, 10000))
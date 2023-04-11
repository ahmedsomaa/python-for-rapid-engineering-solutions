################################################################################
# Function my_sqrt calculates root of a number using Babylonian method         #
# inputs:                                                                      #
# num: the number ro calculate root for                                        #
# guess: the initial guess for the root                                        #
# epislon: the acceptable stop number                                          #
# outputs:                                                                     #
# the root for the give number                                                 #
################################################################################
import numpy as np

def my_sqrt(num, guess, epislon):
    square_guess = guess * guess
    if np.abs(square_guess - num) <= epislon:                         # if value of the guess squred minus the number 
        return guess                                                  # is close to epislon, return the guess
    else:
        num_over_guess = num / guess                                  # if not, then calculate a new value 
        guess = (guess + num_over_guess) / 2                          # for the guess, and re-check again
        return my_sqrt(num, guess, epislon)

EPISLON = 0.01
number = int(input("Enter a number whose square root is desired: "))  # read number to find the root for
initial_guess = int(input("Enter an initial guess: "))                # read initial guess value for the root
square_root = my_sqrt(number, initial_guess, EPISLON)
print("The square root of", number, "is", round(square_root, 2))


# def factorial(n):
#     if n == 1:
#         return 1
#     else:
#         return n * factorial(n - 1)

################################################################################
# Function quad_solver solves quadratic equation                               #
# inputs:                                                                      #
# a, b, c: coefficents of the quadratic equation                               #
# outputs:                                                                     #
# root_1: the first root value that satisfies quadratic equation               #
# root_2: the second root value that satisfies quadratic equation              #
################################################################################
import cmath as cm
import numpy as np

def quad_solver(a, b, c):
    # prepare equation numerator
    square_b = b * b
    four_a_c = 4 * a * c
    discriminant = square_b - four_a_c               # discriminant (b^2 - 4 * a * c) value to determine type of roots

    # prepare equation denominator
    double_a = 2 * a

    if discriminant > 0:                             # roots are real and different
        discriminant_root = np.sqrt(discriminant)
        first_root = (-b + discriminant_root) / double_a
        second_root = (-b - discriminant_root) / double_a
        print("Root 1:", first_root)
        print("Root 2:", second_root)
    elif discriminant == 0:                          # roots are real and equal
        root = -b / double_a
        print("Double root:", root)
    else:                                            # roots are imaginary and unique
        discriminant_root = cm.sqrt(discriminant)
        first_root = (-b + discriminant_root) / double_a
        second_root = (-b - discriminant_root) / double_a
        print("Root 1:", first_root)
        print("Root 2:", second_root)

# read equation coefficients a, b, c as integers
coefficent_a = int(input("Input coefficient a: "))
coefficent_b = int(input("Input coefficient b: "))
coefficent_c = int(input("Input coefficient c: "))

# call quad_solver to print equation roots
quad_solver(coefficent_a, coefficent_b, coefficent_c)
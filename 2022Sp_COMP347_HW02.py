# Name: Maxie Castaneda
# COMP 347 - Machine Learning
# HW No. 2

# Libraries
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from matplotlib import pyplot


# Problem 1 - Linear Regression with Athens Temperature Data
# ------------------------------------------------------------------------------

# In the following problem, implement the solution to the least squares problem.

# 1a. Complete the following functions:


def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
       x: vector of input datas.
       deg: degree of the polynomial fit."""
    A = np.zeros((len(x), deg + 1))  # initialize a matrix full of zeros
    count = deg
    for i in range(deg + 1):
        for j in range(len(x)):
            val = x[j]
            A[j, i] = val ** count
        count -= 1
    return A


def LLS_Solve(x, y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       w = (A'A)-1 A'Y """
    A = A_mat(x, deg)
    AT = A.transpose()
    ATA = np.matmul(AT, A)
    ATAInv = np.linalg.inv(ATA)
    ATY = np.matmul(AT, y)
    w = np.matmul(ATAInv, ATY)
    return w


def LLS_ridge(x, y, deg, lam):
    """Find the vector w that solves the ridge regression problem.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression.
       w = (A'A + LAM*I)-1 A'Y"""
    A = A_mat(x, deg)
    AT = A.transpose()
    ATA = np.matmul(AT, A)
    rows, cols = ATA.shape
    i = np.identity(rows)
    multiplied = i * lam
    ATA_LamI = np.add(ATA, multiplied)
    Inverse = np.linalg.inv(ATA_LamI)
    ATY = np.matmul(AT, y)
    w = np.matmul(Inverse, ATY)
    return w


def poly_func(data, coeffs):
    """ Produce the vector of output data for a polynomial.
        data: x-values of the polynomial.
        coeffs: vector of coefficients for the polynomial.

        what to do:
        data : n x m (rows, columns) --> comes from A matrix
        coeffs : nx1 (data) --> comes from w vector
        y : nx1 (initialize with zeros) --> this is the predicted y values

        for i each row:
            for j in range(m): j should go from 0 to m
                y[i] += data[i,j] * (coeffs[i] raised to columns-j power)"""

    rows, columns = data.shape
    y = np.zeros((rows, 1))
    for i in range(rows):
        for j in range(columns):
            y[i] += data[i, j] * (coeffs[j] ** (columns - j + 1))
    return y


def LLS_func(x, y, w, deg):
    """The linear least squares objective function.
           x: vector of input data.
           y: vector of output data.
           w: vector of weights.
           deg: degree of the polynomial.
           """
    A = A_mat(x, deg)
    AW = np.matmul(A, w)
    norm = np.linalg.norm(np.subtract(AW, y))
    normSq = norm ** 2
    return normSq/len(x)


def RMSE(x, y, w):
    """Compute the root-mean-square error.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       square_root (1/n ((y-AW))^2) """
    A = A_mat(x, len(w) - 1)
    Y_hat = (poly_func(A, w))
    AW = np.matmul(A, w)
    y2 = np.array(y)
    MSE = np.linalg.norm(np.subtract(y, AW)) ** 2
    MSE2 = MSE/len(x)
    RMSE = np.sqrt(MSE2)
    return RMSE


# 1b. Solve the least squares linear regression problem for the Athens
#     temperature data.  Make sure to annotate the plot with the RMSE.

if __name__ == "__main__":
    # 1B:
    data = pd.read_csv('machine learning/athens_ww2_weather.csv')
    x = data['MinTemp']
    y = data['MaxTemp']
    pyplot.scatter(x, y)
    deg1 = 1
    # A_mat(x, deg) --> to get A
    A1 = A_mat(x, deg1)
    print("This is A: ")
    print(A1)
    # LLS_Solve(x, y, deg) --> to get w
    w1 = LLS_Solve(x, y, deg1)
    print("This is w: ")
    print(w1)
    # poly_func(data, coeffs) --> to get y
    # data: x-values of the polynomial.
    # coeffs: vector of coefficients for the polynomial.
    print("This is y: ")
    y1 = poly_func(A1, w1)
    print(y1)
    # LLS_func(x, y, w, deg) --> to get the function
    LLS_function = LLS_func(x, y1, w1, deg1)
    print("This is the solution of the objective function for LLS:")
    print(LLS_function)
    rmse = RMSE(x, y, w1)
    print("This is the rmse: ")
    print(rmse)
    b = w1[1]
    m = w1[0]
    pyplot.title('Athens Temperature')
    pyplot.xlabel('MinTemp', fontsize=15)
    pyplot.ylabel('MaxTemp', fontsize=15)
    pyplot.plot(x, b + m * x, c='r', linewidth=3, alpha=.5, solid_capstyle='round')
    text = "the RMSE is: " + str(rmse)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    pyplot.text(0, 43, text, fontsize=10, verticalalignment='top', bbox=props)
    pyplot.show()

    # Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
    # ------------------------------------------------------------------------------

    # 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with n ranging from 1 to 20.
    #     Additionally, create plots comparing the training error and RMSE for 3 years of data selected at random
    #     (distinct from the years used for training).

    # 1 - choose data to train data
    # I want to do: 1995, 1997, 1994, 1992, 1993

    data2 = pd.read_csv("machine learning/Yosemite_Visits.csv")
    data2 = data2.replace(',', '', regex=True)
    print(data2)
    pre = pd.DataFrame(data2).to_numpy()
    matrix = pre.astype(int)
    matrix = matrix[:, 1:]

    training_1994 = matrix[24]  # 1994
    training_1993 = matrix[25]  # 1993
    training_1992 = matrix[26]  # 1992
    training_1995 = matrix[23]  # 1995
    training_1997 = matrix[21]  # 1997

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_m = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    pyplot.title('Yosemite Visitors with Degree-1 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    # 2 - create model

    y_mat = np.concatenate([training_1994, training_1993, training_1992, training_1995, training_1997])
    # print(training_1994)
    # print(y_mat)

    x_mat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Degree 1
    w_deg1 = LLS_Solve(x_mat, y_mat, 1)
    print("This is w for 1 degree: ")
    print(w_deg1)
    b2 = w_deg1[1]
    m2 = w_deg1[0]
    equation_deg1 = b2 + (m2 * x_m)
    pyplot.plot(x_m, equation_deg1, label='Degree 1 Fit')
    pyplot.legend()
    pyplot.show()

    # Degree 2
    pyplot.title('Yosemite Visitors with Degree-2 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg2 = LLS_Solve(x_mat, y_mat, 2)
    print("This is w for 2 degrees: ")
    print(w_deg2)
    deg2_intercept = w_deg2[2]
    deg_2_first = w_deg2[1]
    deg_2_second = w_deg2[0]
    equation_deg2 = deg2_intercept + (deg_2_first * x_m) + (deg_2_second * x_m ** 2)
    pyplot.plot(x_m, equation_deg2, label='Degree 2 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 3
    pyplot.title('Yosemite Visitors with Degree-3 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg3 = LLS_Solve(x_mat, y_mat, 3)
    print("This is w for 3 degrees: ")
    print(w_deg3)
    deg3_intercept = w_deg3[3]
    deg3_first = w_deg3[2]
    deg3_second = w_deg3[1]
    deg3_third = w_deg3[0]
    equation_deg3 = deg3_intercept + (deg3_first * x_m) + (deg3_second * x_m ** 2) + (deg3_third * x_m ** 3)
    pyplot.plot(x_m, equation_deg3, label='Degree 3 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 4
    pyplot.title('Yosemite Visitors with Degree-4 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg4 = LLS_Solve(x_mat, y_mat, 4)
    print("This is w for 4 degrees: ")
    print(w_deg4)
    deg4_intercept = w_deg4[4]
    deg4_first = w_deg4[3]
    deg4_second = w_deg4[2]
    deg4_third = w_deg4[1]
    deg4_fourth = w_deg4[0]
    equation_deg4 = deg4_intercept + (deg4_first * x_m) + (deg4_second * x_m ** 2) + (deg4_third * x_m ** 3) + (
            deg4_fourth * x_m ** 4)
    pyplot.plot(x_m, equation_deg4, label='Degree 4 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 5
    pyplot.title('Yosemite Visitors with Degree-5 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg5 = LLS_Solve(x_mat, y_mat, 5)
    print("This is w for 5 degrees: ")
    print(w_deg5)
    deg5_intercept = w_deg5[5]
    deg5_first = w_deg5[4]
    deg5_second = w_deg5[3]
    deg5_third = w_deg5[2]
    deg5_fourth = w_deg5[1]
    deg5_fifth = w_deg5[0]
    equation_deg5 = deg5_intercept + (deg5_first * x_m) + (deg5_second * x_m ** 2) + (deg5_third * x_m ** 3) + (
            deg5_fourth * x_m ** 4 + deg5_fifth * x_m ** 5)
    pyplot.plot(x_m, equation_deg5, label='Degree 5 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 6
    pyplot.title('Yosemite Visitors with Degree-6 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg6 = LLS_Solve(x_mat, y_mat, 6)
    print("This is w for 6 degrees: ")
    print(w_deg6)
    deg6_intercept = w_deg6[6]
    deg6_first = w_deg6[5]
    deg6_second = w_deg6[4]
    deg6_third = w_deg6[3]
    deg6_fourth = w_deg6[2]
    deg6_fifth = w_deg6[1]
    deeg6_sixth = w_deg6[0]
    equation_deg6 = deg6_intercept + (deg6_first * x_m) + (deg6_second * x_m ** 2) + (deg6_third * x_m ** 3) + (
            deg6_fourth * x_m ** 4 + deg6_fifth * x_m ** 5 + deeg6_sixth * x_m ** 6)
    pyplot.plot(x_m, equation_deg6, label='Degree 6 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 7
    pyplot.title('Yosemite Visitors with Degree-7 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg7 = LLS_Solve(x_mat, y_mat, 7)
    print("This is w for 7 degrees: ")
    print(w_deg7)
    deg7_intercept = w_deg7[7]
    deg7_first = w_deg7[6]
    deg7_second = w_deg7[5]
    deg7_third = w_deg7[4]
    deg7_fourth = w_deg7[3]
    deg7_fifth = w_deg7[2]
    deg7_sixth = w_deg7[1]
    deg7_seventh = w_deg7[0]
    equation_deg7 = deg7_intercept + (deg7_first * x_m) + (deg7_second * x_m ** 2) + (deg7_third * x_m ** 3) + (
            deg7_fourth * x_m ** 4 + deg7_fifth * x_m ** 5 + deg7_sixth * x_m ** 6 + deg7_seventh * x_m ** 7)
    pyplot.plot(x_m, equation_deg7, label='Degree 7 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 8
    pyplot.title('Yosemite Visitors with Degree-8 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg8 = LLS_Solve(x_mat, y_mat, 8)
    print("This is w for 8 degrees: ")
    print(w_deg8)
    deg8_intercept = w_deg8[8]
    deg8_first = w_deg8[7]
    deg8_second = w_deg8[6]
    deg8_third = w_deg8[5]
    deg8_fourth = w_deg8[4]
    deg8_fifth = w_deg8[3]
    deg8_sixth = w_deg8[2]
    deg8_seventh = w_deg8[1]
    deg8_eighth = w_deg8[0]
    equation_deg8 = deg8_intercept + (deg8_first * x_m) + (deg8_second * x_m ** 2) + (deg8_third * x_m ** 3) + (
            deg8_fourth * x_m ** 4 + deg8_fifth * x_m ** 5 + deg8_sixth * x_m ** 6 + deg8_seventh * x_m ** 7 +
            deg8_eighth * x_m ** 8)
    pyplot.plot(x_m, equation_deg8, label='Degree 8 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 9
    pyplot.title('Yosemite Visitors with Degree-9 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg9 = LLS_Solve(x_mat, y_mat, 9)
    print("This is w for 9 degrees: ")
    print(w_deg9)
    deg9_intercept = w_deg9[9]
    deg9_first = w_deg9[8]
    deg9_second = w_deg9[7]
    deg9_third = w_deg9[6]
    deg9_fourth = w_deg9[5]
    deg9_fifth = w_deg9[4]
    deg9_sixth = w_deg9[3]
    deg9_seventh = w_deg9[2]
    deg9_eighth = w_deg9[1]
    deg9_ninth = w_deg9[0]
    equation_deg9 = deg9_intercept + (deg9_first * x_m) + (deg9_second * x_m ** 2) + (deg9_third * x_m ** 3) + (
            deg9_fourth * x_m ** 4 + deg9_fifth * x_m ** 5 + deg9_sixth * x_m ** 6 + deg9_seventh * x_m ** 7 +
            deg9_eighth * x_m ** 8 + deg9_ninth * x_m ** 9)
    pyplot.plot(x_m, equation_deg9, label='Degree 9 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 10
    pyplot.title('Yosemite Visitors with Degree-10 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg10 = LLS_Solve(x_mat, y_mat, 10)
    print("This is w for 10 degrees: ")
    print(w_deg10)
    deg10_intercept = w_deg10[10]
    deg10_first = w_deg10[9]
    deg10_second = w_deg10[8]
    deg10_third = w_deg10[7]
    deg10_fourth = w_deg10[6]
    deg10_fifth = w_deg10[5]
    deg10_sixth = w_deg10[4]
    deg10_seventh = w_deg10[3]
    deg10_eighth = w_deg10[2]
    deg10_ninth = w_deg10[1]
    deg10_tenth = w_deg10[0]
    equation_deg10 = deg10_intercept + (deg10_first * x_m) + (deg10_second * x_m ** 2) + (deg10_third * x_m ** 3) + (
            deg10_fourth * x_m ** 4 + deg10_fifth * x_m ** 5 + deg10_sixth * x_m ** 6 + deg10_seventh * x_m ** 7 +
            deg10_eighth * x_m ** 8 + deg10_ninth * x_m ** 9 + deg10_tenth * x_m ** 10)
    pyplot.plot(x_m, equation_deg10, label='Degree 10 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 11
    pyplot.title('Yosemite Visitors with Degree-11 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg11 = LLS_Solve(x_mat, y_mat, 11)
    print("This is w for 11 degrees: ")
    print(w_deg11)
    deg11_intercept = w_deg11[11]
    deg11_first = w_deg11[10]
    deg11_second = w_deg11[9]
    deg11_third = w_deg11[8]
    deg11_fourth = w_deg11[7]
    deg11_fifth = w_deg11[6]
    deg11_sixth = w_deg11[5]
    deg11_seventh = w_deg11[4]
    deg11_eighth = w_deg11[3]
    deg11_ninth = w_deg11[2]
    deg11_tenth = w_deg11[1]
    deg11_eleventh = w_deg11[0]
    equation_deg11 = deg11_intercept + (deg11_first * x_m) + (deg11_second * x_m ** 2) + (deg11_third * x_m ** 3) + (
            deg11_fourth * x_m ** 4 + deg11_fifth * x_m ** 5 + deg11_sixth * x_m ** 6 + deg11_seventh * x_m ** 7 +
            deg11_eighth * x_m ** 8 + deg11_ninth * x_m ** 9 + deg11_tenth * x_m ** 10 + deg11_eleventh * x_m ** 11)
    pyplot.plot(x_m, equation_deg11, label='Degree 11 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 12
    pyplot.title('Yosemite Visitors with Degree-12 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg12 = LLS_Solve(x_mat, y_mat, 12)
    print("This is w for 12 degrees: ")
    print(w_deg12)
    deg12_intercept = w_deg12[12]
    deg12_first = w_deg12[11]
    deg12_second = w_deg12[10]
    deg12_third = w_deg12[9]
    deg12_fourth = w_deg12[8]
    deg12_fifth = w_deg12[7]
    deg12_sixth = w_deg12[6]
    deg12_seventh = w_deg12[5]
    deg12_eighth = w_deg12[4]
    deg12_ninth = w_deg12[3]
    deg12_tenth = w_deg12[2]
    deg12_eleventh = w_deg12[1]
    deg12_twelve = w_deg12[0]
    equation_deg12 = deg12_intercept + (deg12_first * x_m) + (deg12_second * x_m ** 2) + (deg12_third * x_m ** 3) + (
            deg12_fourth * x_m ** 4 + deg12_fifth * x_m ** 5 + deg12_sixth * x_m ** 6 + deg12_seventh * x_m ** 7 +
            deg12_eighth * x_m ** 8 + deg12_ninth * x_m ** 9 + deg12_tenth * x_m ** 10 + deg12_eleventh * x_m ** 11 +
            deg12_twelve * x_m ** 12)
    pyplot.plot(x_m, equation_deg12, label='Degree 12 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 13
    pyplot.title('Yosemite Visitors with Degree-13 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg13 = LLS_Solve(x_mat, y_mat, 13)
    print("This is w for 13 degrees: ")
    print(w_deg13)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)
    pyplot.plot(x_m, equation_deg13, label='Degree 13 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 14
    pyplot.title('Yosemite Visitors with Degree-14 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg14 = LLS_Solve(x_mat, y_mat, 14)
    print("This is w for 14 degrees: ")
    print(w_deg14)
    deg14_intercept = w_deg14[14]
    deg14_first = w_deg14[13]
    deg14_second = w_deg14[12]
    deg14_third = w_deg14[11]
    deg14_fourth = w_deg14[10]
    deg14_fifth = w_deg14[9]
    deg14_sixth = w_deg14[8]
    deg14_seventh = w_deg14[7]
    deg14_eighth = w_deg14[6]
    deg14_ninth = w_deg14[5]
    deg14_tenth = w_deg14[4]
    deg14_eleventh = w_deg14[3]
    deg14_twelve = w_deg14[2]
    deg14_thirteen = w_deg14[1]
    deg14_fourteen = w_deg14[0]
    equation_deg14 = deg14_intercept + (deg14_first * x_m) + (deg14_second * x_m ** 2) + (deg14_third * x_m ** 3) + (
            deg14_fourth * x_m ** 4 + deg14_fifth * x_m ** 5 + deg14_sixth * x_m ** 6 + deg14_seventh * x_m ** 7 +
            deg14_eighth * x_m ** 8 + deg14_ninth * x_m ** 9 + deg14_tenth * x_m ** 10 + deg14_eleventh * x_m ** 11 +
            deg14_twelve * x_m ** 12 + deg14_thirteen * x_m ** 13 + deg14_fourteen * x_m ** 14)
    pyplot.plot(x_m, equation_deg14, label='Degree 14 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 15
    pyplot.title('Yosemite Visitors with Degree-15 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg15 = LLS_Solve(x_mat, y_mat, 15)
    print("This is w for 15 degrees: ")
    print(w_deg15)
    deg15_intercept = w_deg15[15]
    deg15_first = w_deg15[14]
    deg15_second = w_deg15[13]
    deg15_third = w_deg15[12]
    deg15_fourth = w_deg15[11]
    deg15_fifth = w_deg15[10]
    deg15_sixth = w_deg15[9]
    deg15_seventh = w_deg15[8]
    deg15_eighth = w_deg15[7]
    deg15_ninth = w_deg15[6]
    deg15_tenth = w_deg15[5]
    deg15_eleventh = w_deg15[4]
    deg15_twelve = w_deg15[3]
    deg15_thirteen = w_deg15[2]
    deg15_fourteen = w_deg15[1]
    deg15_fifteen = w_deg15[0]
    equation_deg15 = deg15_intercept + (deg15_first * x_m) + (deg15_second * x_m ** 2) + (deg15_third * x_m ** 3) + (
            deg15_fourth * x_m ** 4 + deg15_fifth * x_m ** 5 + deg15_sixth * x_m ** 6 + deg15_seventh * x_m ** 7 +
            deg15_eighth * x_m ** 8 + deg15_ninth * x_m ** 9 + deg15_tenth * x_m ** 10 + deg15_eleventh * x_m ** 11 +
            deg15_twelve * x_m ** 12 + deg15_thirteen * x_m ** 13 + deg15_fourteen * x_m ** 14 + deg15_fifteen * x_m ** 15)
    pyplot.plot(x_m, equation_deg15, label='Degree 15 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 16
    pyplot.title('Yosemite Visitors with Degree-16 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg16 = LLS_Solve(x_mat, y_mat, 16)
    print("This is w for 16 degrees: ")
    print(w_deg16)
    deg16_intercept = w_deg16[16]
    deg16_first = w_deg16[15]
    deg16_second = w_deg16[14]
    deg16_third = w_deg16[13]
    deg16_fourth = w_deg16[12]
    deg16_fifth = w_deg16[11]
    deg16_sixth = w_deg16[10]
    deg16_seventh = w_deg16[9]
    deg16_eighth = w_deg16[8]
    deg16_ninth = w_deg16[7]
    deg16_tenth = w_deg16[6]
    deg16_eleventh = w_deg16[5]
    deg16_twelve = w_deg16[4]
    deg16_thirteen = w_deg16[3]
    deg16_fourteen = w_deg16[2]
    deg16_fifteen = w_deg16[1]
    deg16_sixteen = w_deg16[0]
    equation_deg16 = deg16_intercept + (deg16_first * x_m) + (deg16_second * x_m ** 2) + (deg16_third * x_m ** 3) + (
            deg16_fourth * x_m ** 4 + deg16_fifth * x_m ** 5 + deg16_sixth * x_m ** 6 + deg16_seventh * x_m ** 7 +
            deg16_eighth * x_m ** 8 + deg16_ninth * x_m ** 9 + deg16_tenth * x_m ** 10 + deg16_eleventh * x_m ** 11 +
            deg16_twelve * x_m ** 12 + deg16_thirteen * x_m ** 13 + deg16_fourteen * x_m ** 14 + deg16_fifteen * x_m ** 15
            + deg16_sixteen * x_m ** 16)
    pyplot.plot(x_m, equation_deg16, label='Degree 16 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 17
    pyplot.title('Yosemite Visitors with Degree-17 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg17 = LLS_Solve(x_mat, y_mat, 17)
    print("This is w for 17 degrees: ")
    print(w_deg17)
    deg17_intercept = w_deg17[17]
    deg17_first = w_deg17[16]
    deg17_second = w_deg17[15]
    deg17_third = w_deg17[14]
    deg17_fourth = w_deg17[13]
    deg17_fifth = w_deg17[12]
    deg17_sixth = w_deg17[11]
    deg17_seventh = w_deg17[10]
    deg17_eighth = w_deg17[9]
    deg17_ninth = w_deg17[8]
    deg17_tenth = w_deg17[7]
    deg17_eleventh = w_deg17[6]
    deg17_twelve = w_deg17[5]
    deg17_thirteen = w_deg17[4]
    deg17_fourteen = w_deg17[3]
    deg17_fifteen = w_deg17[2]
    deg17_sixteen = w_deg17[1]
    deg17_seventeen = w_deg17[0]
    equation_deg17 = deg17_intercept + (deg17_first * x_m) + (deg17_second * x_m ** 2) + (deg17_third * x_m ** 3) + (
            deg17_fourth * x_m ** 4 + deg17_fifth * x_m ** 5 + deg17_sixth * x_m ** 6 + deg17_seventh * x_m ** 7 +
            deg17_eighth * x_m ** 8 + deg17_ninth * x_m ** 9 + deg17_tenth * x_m ** 10 + deg17_eleventh * x_m ** 11 +
            deg17_twelve * x_m ** 12 + deg17_thirteen * x_m ** 13 + deg17_fourteen * x_m ** 14 + deg17_fifteen * x_m ** 15
            + deg17_sixteen * x_m ** 16 + deg17_seventeen * x_m ** 17)
    pyplot.plot(x_m, equation_deg17, label='Degree 17 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 18
    pyplot.title('Yosemite Visitors with Degree-18 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg18 = LLS_Solve(x_mat, y_mat, 18)
    print("This is w for 18 degrees: ")
    print(w_deg18)
    deg18_intercept = w_deg18[18]
    deg18_first = w_deg18[17]
    deg18_second = w_deg18[16]
    deg18_third = w_deg18[15]
    deg18_fourth = w_deg18[14]
    deg18_fifth = w_deg18[13]
    deg18_sixth = w_deg18[12]
    deg18_seventh = w_deg18[11]
    deg18_eighth = w_deg18[10]
    deg18_ninth = w_deg18[9]
    deg18_tenth = w_deg18[8]
    deg18_eleventh = w_deg18[7]
    deg18_twelve = w_deg18[6]
    deg18_thirteen = w_deg18[5]
    deg18_fourteen = w_deg18[4]
    deg18_fifteen = w_deg18[3]
    deg18_sixteen = w_deg18[2]
    deg18_seventeen = w_deg18[1]
    deg18_eighteen = w_deg18[0]
    equation_deg18 = deg18_intercept + (deg18_first * x_m) + (deg18_second * x_m ** 2) + (deg18_third * x_m ** 3) + (
            deg18_fourth * x_m ** 4 + deg18_fifth * x_m ** 5 + deg18_sixth * x_m ** 6 + deg18_seventh * x_m ** 7 +
            deg18_eighth * x_m ** 8 + deg18_ninth * x_m ** 9 + deg18_tenth * x_m ** 10 + deg18_eleventh * x_m ** 11 +
            deg18_twelve * x_m ** 12 + deg18_thirteen * x_m ** 13 + deg18_fourteen * x_m ** 14 + deg18_fifteen * x_m ** 15
            + deg18_sixteen * x_m ** 16 + deg18_seventeen * x_m ** 17 + deg18_eighteen * x_m ** 18)
    pyplot.plot(x_m, equation_deg18, label='Degree 18 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 19
    pyplot.title('Yosemite Visitors with Degree-19 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg19 = LLS_Solve(x_mat, y_mat, 19)
    print("This is w for 19 degrees: ")
    print(w_deg19)
    deg19_intercept = w_deg19[19]
    deg19_first = w_deg19[18]
    deg19_second = w_deg19[17]
    deg19_third = w_deg19[16]
    deg19_fourth = w_deg19[15]
    deg19_fifth = w_deg19[14]
    deg19_sixth = w_deg19[13]
    deg19_seventh = w_deg19[12]
    deg19_eighth = w_deg19[11]
    deg19_ninth = w_deg19[10]
    deg19_tenth = w_deg19[9]
    deg19_eleventh = w_deg19[8]
    deg19_twelve = w_deg19[7]
    deg19_thirteen = w_deg19[6]
    deg19_fourteen = w_deg19[5]
    deg19_fifteen = w_deg19[4]
    deg19_sixteen = w_deg19[3]
    deg19_seventeen = w_deg19[2]
    deg19_eighteen = w_deg19[1]
    deg19_nineteen = w_deg19[0]
    equation_deg19 = deg19_intercept + (deg19_first * x_m) + (deg19_second * x_m ** 2) + (deg19_third * x_m ** 3) + (
            deg19_fourth * x_m ** 4 + deg19_fifth * x_m ** 5 + deg19_sixth * x_m ** 6 + deg19_seventh * x_m ** 7 +
            deg19_eighth * x_m ** 8 + deg19_ninth * x_m ** 9 + deg19_tenth * x_m ** 10 + deg19_eleventh * x_m ** 11 +
            deg19_twelve * x_m ** 12 + deg19_thirteen * x_m ** 13 + deg19_fourteen * x_m ** 14 + deg19_fifteen * x_m ** 15
            + deg19_sixteen * x_m ** 16 + deg19_seventeen * x_m ** 17 + deg19_eighteen * x_m ** 18 + deg19_nineteen * x_m ** 19)
    pyplot.plot(x_m, equation_deg19, label='Degree 19 Fit')

    pyplot.legend()
    pyplot.show()

    # Degree 20
    pyplot.title('Yosemite Visitors with Degree-20 LL')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')

    w_deg20 = LLS_Solve(x_mat, y_mat, 20)
    print("This is w for 20 degrees: ")
    print(w_deg20)
    deg20_intercept = w_deg20[20]
    deg20_first = w_deg20[19]
    deg20_second = w_deg20[18]
    deg20_third = w_deg20[17]
    deg20_fourth = w_deg20[16]
    deg20_fifth = w_deg20[15]
    deg20_sixth = w_deg20[14]
    deg20_seventh = w_deg20[13]
    deg20_eighth = w_deg20[12]
    deg20_ninth = w_deg20[11]
    deg20_tenth = w_deg20[10]
    deg20_eleventh = w_deg20[9]
    deg20_twelve = w_deg20[8]
    deg20_thirteen = w_deg20[7]
    deg20_fourteen = w_deg20[6]
    deg20_fifteen = w_deg20[5]
    deg20_sixteen = w_deg20[4]
    deg20_seventeen = w_deg20[3]
    deg20_eighteen = w_deg20[2]
    deg20_nineteen = w_deg20[1]
    deg20_twenty = w_deg20[0]
    equation_deg20 = deg20_intercept + (deg20_first * x_m) + (deg20_second * x_m ** 2) + (deg20_third * x_m ** 3) + (
            deg20_fourth * x_m ** 4 + deg20_fifth * x_m ** 5 + deg20_sixth * x_m ** 6 + deg20_seventh * x_m ** 7 +
            deg20_eighth * x_m ** 8 + deg20_ninth * x_m ** 9 + deg20_tenth * x_m ** 10 + deg20_eleventh * x_m ** 11 +
            deg20_twelve * x_m ** 12 + deg20_thirteen * x_m ** 13 + deg20_fourteen * x_m ** 14 + deg20_fifteen * x_m ** 15
            + deg20_sixteen * x_m ** 16 + deg20_seventeen * x_m ** 17 + deg20_eighteen * x_m ** 18 + deg20_nineteen * x_m ** 19
            + deg20_twenty * x_m ** 20)
    pyplot.plot(x_m, equation_deg20, label='Degree 20 Fit')

    pyplot.legend()
    pyplot.show()

    # Additionally, create plots comparing the training error and
    # RMSE for 3 years of data selected at random
    # (distinct from the years used for training).
    # 4- compare
    # training Error = LLS solve
    # 3 - insert 3 years randomly
    test_2018 = matrix[0]
    test_2011 = matrix[7]
    test_2007 = matrix[11]

    y_matTest = test_2018
    y_matTest2 = test_2011
    y_matTest3 = test_2007

    x_matTest = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    rsme_test = [RMSE(x_matTest, y_matTest, w_deg1), RMSE(x_matTest, y_matTest, w_deg2),
                 RMSE(x_matTest, y_matTest, w_deg3), RMSE(x_matTest, y_matTest, w_deg4),
                 RMSE(x_matTest, y_matTest, w_deg5), RMSE(x_matTest, y_matTest, w_deg6),
                 RMSE(x_matTest, y_matTest, w_deg7), RMSE(x_matTest, y_matTest, w_deg8),
                 RMSE(x_matTest, y_matTest, w_deg9), RMSE(x_matTest, y_matTest, w_deg10),
                 RMSE(x_matTest, y_matTest, w_deg11), RMSE(x_matTest, y_matTest, w_deg12),
                 RMSE(x_matTest, y_matTest, w_deg13), RMSE(x_matTest, y_matTest, w_deg14),
                 RMSE(x_matTest, y_matTest, w_deg15), RMSE(x_matTest, y_matTest, w_deg16),
                 RMSE(x_matTest, y_matTest, w_deg17), RMSE(x_matTest, y_matTest, w_deg18),
                 RMSE(x_matTest, y_matTest, w_deg19), RMSE(x_matTest, y_matTest, w_deg20)]

    rsme_test2 = [RMSE(x_matTest, y_matTest2, w_deg1), RMSE(x_matTest, y_matTest2, w_deg2),
                 RMSE(x_matTest, y_matTest2, w_deg3), RMSE(x_matTest, y_matTest2, w_deg4),
                 RMSE(x_matTest, y_matTest2, w_deg5), RMSE(x_matTest, y_matTest2, w_deg6),
                 RMSE(x_matTest, y_matTest2, w_deg7), RMSE(x_matTest, y_matTest2, w_deg8),
                 RMSE(x_matTest, y_matTest2, w_deg9), RMSE(x_matTest, y_matTest2, w_deg10),
                 RMSE(x_matTest, y_matTest2, w_deg11), RMSE(x_matTest, y_matTest2, w_deg12),
                 RMSE(x_matTest, y_matTest2, w_deg13), RMSE(x_matTest, y_matTest2, w_deg14),
                 RMSE(x_matTest, y_matTest2, w_deg15), RMSE(x_matTest, y_matTest2, w_deg16),
                 RMSE(x_matTest, y_matTest2, w_deg17), RMSE(x_matTest, y_matTest2, w_deg18),
                 RMSE(x_matTest, y_matTest2, w_deg19), RMSE(x_matTest, y_matTest2, w_deg20)]

    rsme_test3 = [RMSE(x_matTest, y_matTest3, w_deg1), RMSE(x_matTest, y_matTest3, w_deg2),
                 RMSE(x_matTest, y_matTest3, w_deg3), RMSE(x_matTest, y_matTest3, w_deg4),
                 RMSE(x_matTest, y_matTest3, w_deg5), RMSE(x_matTest, y_matTest3, w_deg6),
                 RMSE(x_matTest, y_matTest3, w_deg7), RMSE(x_matTest, y_matTest3, w_deg8),
                 RMSE(x_matTest, y_matTest3, w_deg9), RMSE(x_matTest, y_matTest3, w_deg10),
                 RMSE(x_matTest, y_matTest3, w_deg11), RMSE(x_matTest, y_matTest3, w_deg12),
                 RMSE(x_matTest, y_matTest3, w_deg13), RMSE(x_matTest, y_matTest3, w_deg14),
                 RMSE(x_matTest, y_matTest3, w_deg15), RMSE(x_matTest, y_matTest3, w_deg16),
                 RMSE(x_matTest, y_matTest3, w_deg17), RMSE(x_matTest, y_matTest3, w_deg18),
                 RMSE(x_matTest, y_matTest3, w_deg19), RMSE(x_matTest, y_matTest3, w_deg20)]

    pyplot.title('Yosemite Visitors RMSE')
    pyplot.xlabel('Degrees', fontsize=10)
    ax = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    pyplot.xticks(ax, ax)
    pyplot.plot(ax, rsme_test, label="2018")
    pyplot.plot(ax, rsme_test2, label="2011")
    pyplot.plot(ax, rsme_test3, label="2007")
    pyplot.legend()
    pyplot.show()

    training_error = [LLS_func(x_matTest, y_matTest, w_deg1, 1), LLS_func(x_matTest, y_matTest, w_deg2, 2),
                      LLS_func(x_matTest, y_matTest, w_deg3, 3), LLS_func(x_matTest, y_matTest, w_deg4, 4),
                      LLS_func(x_matTest, y_matTest, w_deg5, 5), LLS_func(x_matTest, y_matTest, w_deg6, 6),
                      LLS_func(x_matTest, y_matTest, w_deg7, 7), LLS_func(x_matTest, y_matTest, w_deg8, 8),
                      LLS_func(x_matTest, y_matTest, w_deg9, 9), LLS_func(x_matTest, y_matTest, w_deg10, 10),
                      LLS_func(x_matTest, y_matTest, w_deg11, 11), LLS_func(x_matTest, y_matTest, w_deg12, 12),
                      LLS_func(x_matTest, y_matTest, w_deg13, 13), LLS_func(x_matTest, y_matTest, w_deg14, 14),
                      LLS_func(x_matTest, y_matTest, w_deg15, 15), LLS_func(x_matTest, y_matTest, w_deg16, 16),
                      LLS_func(x_matTest, y_matTest, w_deg17, 17), LLS_func(x_matTest, y_matTest, w_deg18, 18),
                      LLS_func(x_matTest, y_matTest, w_deg19, 19), LLS_func(x_matTest, y_matTest, w_deg20, 20)]

    training_error2 = [LLS_func(x_matTest, y_matTest2, w_deg1, 1), LLS_func(x_matTest, y_matTest2, w_deg2, 2),
                      LLS_func(x_matTest, y_matTest2, w_deg3, 3), LLS_func(x_matTest, y_matTest2, w_deg4, 4),
                      LLS_func(x_matTest, y_matTest2, w_deg5, 5), LLS_func(x_matTest, y_matTest2, w_deg6, 6),
                      LLS_func(x_matTest, y_matTest2, w_deg7, 7), LLS_func(x_matTest, y_matTest2, w_deg8, 8),
                      LLS_func(x_matTest, y_matTest2, w_deg9, 9), LLS_func(x_matTest, y_matTest2, w_deg10, 10),
                      LLS_func(x_matTest, y_matTest2, w_deg11, 11), LLS_func(x_matTest, y_matTest2, w_deg12, 12),
                      LLS_func(x_matTest, y_matTest2, w_deg13, 13), LLS_func(x_matTest, y_matTest2, w_deg14, 14),
                      LLS_func(x_matTest, y_matTest2, w_deg15, 15), LLS_func(x_matTest, y_matTest2, w_deg16, 16),
                      LLS_func(x_matTest, y_matTest2, w_deg17, 17), LLS_func(x_matTest, y_matTest2, w_deg18, 18),
                      LLS_func(x_matTest, y_matTest2, w_deg19, 19), LLS_func(x_matTest, y_matTest2, w_deg20, 20)]

    training_error3 = [LLS_func(x_matTest, y_matTest3, w_deg1, 1), LLS_func(x_matTest, y_matTest3, w_deg2, 2),
                      LLS_func(x_matTest, y_matTest3, w_deg3, 3), LLS_func(x_matTest, y_matTest3, w_deg4, 4),
                      LLS_func(x_matTest, y_matTest3, w_deg5, 5), LLS_func(x_matTest, y_matTest3, w_deg6, 6),
                      LLS_func(x_matTest, y_matTest3, w_deg7, 7), LLS_func(x_matTest, y_matTest3, w_deg8, 8),
                      LLS_func(x_matTest, y_matTest3, w_deg9, 9), LLS_func(x_matTest, y_matTest3, w_deg10, 10),
                      LLS_func(x_matTest, y_matTest3, w_deg11, 11), LLS_func(x_matTest, y_matTest3, w_deg12, 12),
                      LLS_func(x_matTest, y_matTest3, w_deg13, 13), LLS_func(x_matTest, y_matTest3, w_deg14, 14),
                      LLS_func(x_matTest, y_matTest3, w_deg15, 15), LLS_func(x_matTest, y_matTest3, w_deg16, 16),
                      LLS_func(x_matTest, y_matTest3, w_deg17, 17), LLS_func(x_matTest, y_matTest3, w_deg18, 18),
                      LLS_func(x_matTest, y_matTest3, w_deg19, 19), LLS_func(x_matTest, y_matTest3, w_deg20, 20)]

    pyplot.title('Yosemite Visitors Training Error')
    pyplot.xlabel('Degrees', fontsize=10)
    ax = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    pyplot.xticks(ax, ax)
    pyplot.plot(ax, training_error, label="2018")
    pyplot.plot(ax, training_error2, label="2011")
    pyplot.plot(ax, training_error3, label="2007")
    pyplot.legend()
    pyplot.show()

# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.
#     six degrees
#     Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.
#     lamda = .05 +

    y_mat = np.concatenate([training_1994, training_1993, training_1992, training_1995, training_1997])
    x_mat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_m = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.05)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.05', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.1)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.1', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.15)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.15', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.2)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.2', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.25)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.25', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.3)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.3', linewidth=5)
    pyplot.legend()
    pyplot.show()


    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.35)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.35', linewidth=5)
    pyplot.legend()
    pyplot.show()



    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.4)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.4', linewidth=5)
    pyplot.legend()
    pyplot.show()


    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.45)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.45', linewidth=5)
    pyplot.legend()
    pyplot.show()



    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.5)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.5', linewidth=5)
    pyplot.legend()
    pyplot.show()


    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.55)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.55', linewidth=5)
    pyplot.legend()
    pyplot.show()


    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.6)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.6', linewidth=5)
    pyplot.legend()
    pyplot.show()


    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.65)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.65', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.7)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.7', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.75)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.75', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.8)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.8', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.85)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.85', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.9)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.9', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 0.95)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 0.95', linewidth=5)
    pyplot.legend()
    pyplot.show()

    w_deg13 = LLS_ridge(x_mat, y_mat, 13, 1)
    deg13_intercept = w_deg13[13]
    deg13_first = w_deg13[12]
    deg13_second = w_deg13[11]
    deg13_third = w_deg13[10]
    deg13_fourth = w_deg13[9]
    deg13_fifth = w_deg13[8]
    deg13_sixth = w_deg13[7]
    deg13_seventh = w_deg13[6]
    deg13_eighth = w_deg13[5]
    deg13_ninth = w_deg13[4]
    deg13_tenth = w_deg13[3]
    deg13_eleventh = w_deg13[2]
    deg13_twelve = w_deg13[1]
    deg13_thirteen = w_deg13[0]
    equation_deg13 = deg13_intercept + (deg13_first * x_m) + (deg13_second * x_m ** 2) + (deg13_third * x_m ** 3) + (
            deg13_fourth * x_m ** 4 + deg13_fifth * x_m ** 5 + deg13_sixth * x_m ** 6 + deg13_seventh * x_m ** 7 +
            deg13_eighth * x_m ** 8 + deg13_ninth * x_m ** 9 + deg13_tenth * x_m ** 10 + deg13_eleventh * x_m ** 11 +
            deg13_twelve * x_m ** 12 + deg13_thirteen * x_m ** 13)

    pyplot.title('Yosemite Visitors Ridge Reg with Degree 13')
    pyplot.xlabel('Months', fontsize=10)
    pyplot.ylabel('Num of Visitors', fontsize=8)
    pyplot.xticks(x_m, months)
    pyplot.plot(x_m, training_1997, label='1997')
    pyplot.plot(x_m, training_1994, label='1994')
    pyplot.plot(x_m, training_1995, label='1995')
    pyplot.plot(x_m, training_1992, label='1992')
    pyplot.plot(x_m, training_1993, label='1993')
    pyplot.plot(x_m, equation_deg13, label='Lambda = 1', linewidth=5)
    pyplot.legend()
    pyplot.show()




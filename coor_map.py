from math import *
import numpy as np


def transform(xp, yp, xc, yc, theta):
    trans_matrix = np.array([[cos(theta), -sin(theta), xp],
                            [sin(theta), cos(theta), yp],
                            [0, 0, 1]])
    car_coord = np.array([[xc], [yc], [1]])

    result = trans_matrix.dot(car_coord)
    print(result)


def multivariate_gaussian(x, y, mu_x, mu_y, sig_x, sig_y):
    gauss_norm = (1 / (2 * np.pi * sig_x * sig_y))
    exponent = -(((x - mu_x) ** 2) / (2 * sig_x ** 2) + ((y - mu_y) ** 2) / (2 * sig_y ** 2))
    #exponent = -((x-mu_x)**2) / 0.18 + ((y-mu_y)**2) / 0.18
    prob = np.exp(exponent) * gauss_norm

    print(prob)
    return prob



# transform(4, 5, 2, 2, -pi/2)
# transform(4, 5, 3, -2, -pi/2)
# transform(4, 5, 0, -4, -pi/2)

p1 = multivariate_gaussian(6, 3, 5, 3, 0.3, 0.3)
p2 = multivariate_gaussian(2, 2, 2, 1, 0.3, 0.3)
p3 = multivariate_gaussian(0, 5, 2, 1, 0.3, 0.3)

print(p1*p2*p3)
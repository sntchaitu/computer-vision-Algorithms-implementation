__author__ = 'chaitanya'
#Below program will print the determinant  if determinant is non-zero else it will print "matrix is singular"
import numpy as np


def compute_determinant(mat):
    """
        computes the determinant of 3X3 matrix and returns the value
    """
    """
    :input mat
    :return:number
    """

    value1 = (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])) - (
    mat[0][1] * (mat[2][2] * mat[1][0] - mat[1][2] * mat[2][0])) + (
             mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]))

    return value1

# frames 3X3 matrix with each value from 0 to 1 using random.random function
mat1 = np.random.random((3, 3))

value = compute_determinant(mat1)

# if the determinant value is zero then prints the matrix is singular else print the determinant value

if value == 0:
    print "matrix is singular"
else:

    print value

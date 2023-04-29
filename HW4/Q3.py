import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('L.txt', header = None, sep = " ")

# print(data)

arr = np.array(data)
print(arr)

eigenvalues, eigenvector = np.linalg.eig(arr)

min_val = np.min(eigenvalues[np.nonzero(eigenvalues)])

min_index = np.where(eigenvalues == min_val)[0][0]

print("eigenvalues for L: ", eigenvalues)
print("Smallest non-zero value:", min_val)
print("Index of smallest non-zero value:", min_index)
print()
print("eigenvector values for L: ", eigenvector[min_index])

U = eigenvector[min_index]

print()
print()
print()
print()
print()

data1 = pd.read_csv('Ls.txt', header = None, sep = " ")

arr1 = np.array(data1)
eigenvalues1, eigenvectors1 = np.linalg.eig(arr1)

min_val = np.min(eigenvalues1[np.nonzero(eigenvalues1)])

min_index = np.where(eigenvalues1 == min_val)[0][0]

print("eigenvalues for Ls : ", eigenvalues1)
print("Smallest non-zero value:", min_val)
print("Index of smallest non-zero value:", min_index)
print()
print("eigenvector values for Ls: ", eigenvectors1[min_index])

U_s = eigenvectors1[min_index]

node_id = [1,2,3,4,5,6]
#
# plt.plot(node_id, U, label = "values for L")
# plt.plot(node_id, U_s, label = "values for Ls")
#
# plt.xlabel('node_id')
# plt.ylabel('eigenvector values')
# plt.title('3B question')
#
# # add a legend
# plt.legend()

# display the graph
# plt.show()

import math

def eigencalc( i, j):
    dict = {}
    for check in range(0, len(U)):
        if (j == len(U)):
            break;
        dict[i + 1, j + 1] = math.sqrt(math.pow(((j + 1) - (i + 1)), 2) + math.pow((U[i] - U[j]), 2))
        j = j + 1
    return dict

def euclidcalc(val, val2, x, y):
    return math.sqrt(math.pow((y-x),2) + math.pow((val2-val),2))


print("for point 1", eigencalc(0,1))
print("for point 2", eigencalc(1,2))
print("for point 3", eigencalc(2,3))
print("for point 4", eigencalc(3,4))
print("for point 5", eigencalc(4,5))


def eigencalc_s( i, j):
    dict = {}
    for check in range(0, len(U_s)):
        if (j == len(U_s)):
            break;
        dict[i + 1, j + 1] = math.sqrt(math.pow(((j + 1) - (i + 1)), 2) + math.pow((U_s[i] - U_s[j]), 2))
        j = j + 1
    return dict

print()
print(" -----------------------------------------------")

print("for point 1", eigencalc_s(0,1))
print("for point 2", eigencalc_s(1,2))
print("for point 3", eigencalc_s(2,3))
print("for point 4", eigencalc_s(3,4))
print("for point 5", eigencalc_s(4,5))
import math

import numpy as np


data = [[1,0,1],[0,0,1],[2,3,-1],[2,2,-1]]

a = np.array(data)
print(a)

labels = a[:, 2]
print(labels)

w1 = np.array([1,1])
w2 = np.array([1,0])
print(w1,w2)

set1 =  (a[labels == 1, :1] +  a[labels == 1, 1:2])/2
print(np.transpose(w2)/math.sqrt(2))

mean1 = np.dot(w1.T/math.sqrt(2) , set1)
mean2 = np.dot(w2.T/math.sqrt(1), set1)

print("mean of class1 w1 w2: ", mean1, mean2)
print("variance: ")


import statistics

import numpy as np
import pandas as pd
import pickle

file = open('cloud.data', 'r')
data = []
data.append(file.read())
file.close()

word = data[0].split()
# print(word)
data = []
for i in word:
    data.append(float(i))
# print(data)

data = np.array(data)

center = lambda x:x-x.mean()

data_aftercenter = center(data)

print(list(data_aftercenter))
print(data_aftercenter.mean())


import math

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [10.50, 4.50]
plt.rcParams["figure.autolayout"] = True

dict_data = {0: 60, 1:720, 2:1200, 3:750, 4:240, 10:30}
prob = []
for i in list(dict_data.keys()):
    prob.append(dict_data.get(i)/3000)
print(prob)
for i in range(len(prob)):
    if i == 0: prob[i] = prob[i]
    else: prob[i] = prob[i] + prob[i-1]
print(prob)
data= np.array(prob)
x = np.array([0,1,2,3,4,10,11])
print(x)
print(data)
print(data[5])
fig, ax = plt.subplots()
# ax.plot(x,data)
#--line 1---
ax.hlines(y = data[0], xmin = x[0], xmax=x[1])
ax.plot(x[0],data[0], marker='o', color='black')
ax.plot(x[1],data[0], marker='o', markerfacecolor='white', markeredgecolor='black')
#--line 2--
ax.hlines(y = data[1], xmin = x[1], xmax=x[2])
ax.plot(x[1],data[1], marker='o', color='black')
ax.plot(x[2],data[1], marker='o', markerfacecolor='white', markeredgecolor='black')

#--line 3---
ax.hlines(y = data[2], xmin = x[2], xmax=x[3])
ax.plot(x[2],data[2], marker='o', color='black')
ax.plot(x[3],data[2], marker='o', markerfacecolor='white', markeredgecolor='black')

#--line 4---
ax.hlines(y = data[3], xmin = x[3], xmax=x[4])
ax.plot(x[3],data[3], marker='o', color='black')
ax.plot(x[4],data[3], marker='o', markerfacecolor='white', markeredgecolor='black')

#--line 5---
ax.hlines(y = data[4], xmin = x[4], xmax=x[5])
ax.plot(x[4],data[4], marker='o', color='black')
ax.plot(x[5],data[4], marker='o', markerfacecolor='white', markeredgecolor='black')

#--line 6---
ax.hlines(y = data[5], xmin=x[5], xmax=x[6])
ax.plot(x[5],data[5], marker='o', color='black')
ax.plot(x[6],data[5], marker='o', markerfacecolor='white', markeredgecolor='black')

plt.show()
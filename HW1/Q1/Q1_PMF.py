import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [10.50, 4.50]
plt.rcParams["figure.autolayout"] = True

dict_data = {0: 60, 1:720, 2:1200, 3:750, 4:240, 10:30}
prob = []
for i in list(dict_data.keys()):
    prob.append(dict_data.get(i)/3000)
print(prob)
data= np.array(prob)
x = np.array([0,1,2,3,4,10,11])
print(x)
print(data)

fig, ax = plt.subplots()
# ax.plot(x,data)

#--line 1---
ax.vlines(x = x[0], ymin = 0,ymax = data[0])
# ax.plot(x[0],data[0], marker='o', color='black')
ax.plot(x[0],data[0], marker='o', markerfacecolor='white', markeredgecolor='black')

#--line 2--
ax.vlines(x = x[1], ymin = 0,ymax = data[1])
ax.plot(x[1],data[1], marker='o', markerfacecolor='white', markeredgecolor='black')
# #--line 3---
ax.vlines(x = x[2], ymin = 0,ymax = data[2])
ax.plot(x[2],data[2], marker='o', markerfacecolor='white', markeredgecolor='black')
# #--line 4---
ax.vlines(x = x[3], ymin = 0,ymax = data[3])
ax.plot(x[3],data[3], marker='o', markerfacecolor='white', markeredgecolor='black')
# #--line 5---
ax.vlines(x = x[4], ymin = 0,ymax = data[4])
ax.plot(x[4],data[4], marker='o', markerfacecolor='white', markeredgecolor='black')
# #--line 6---
ax.vlines(x = x[5], ymin = 0,ymax = data[5])
ax.plot(x[5],data[5], marker='o', markerfacecolor='white', markeredgecolor='black')


plt.show()
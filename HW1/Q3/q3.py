import numpy as np
import pandas as pd
import re
from numpy.linalg import eig
import matplotlib.pyplot as plt

df = pd.read_csv("cloud.data", header=None)
WS = re.compile(r"\s+")
df = df[0].apply(lambda x:pd.to_numeric(pd.Series(WS.sub("  ",x).split("  ")),  errors='coerce'))

df_centered = df.apply(lambda column: column-column.mean())
# print(df.cov())

def cov(x,y):
    x = x-x.mean()
    y = y-y.mean()
    cov_val = 0
    for i in range(0,1024):
        cov_val += float(x[i])*float(y[i])
    return cov_val/1024

# print("covariance:", cov(df[df.columns[0]], df[df.columns[7]]))
def covariance(df):
    covariance_matrix = {}
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if (i == j):
                covariance_matrix[i, j] = float(df[df.columns[i]].var())
            else:
                covariance_matrix[i, j] = cov(df[df.columns[i]], df[df.columns[j]])
    return covariance_matrix


covariance_matrix = covariance(df)
covariance_array = []
for i in range(0,10):
    current = []
    for j in range(0,10):
        current.append(covariance_matrix.get((i,j)))
    covariance_array.append(current)
print("the covariance matrix with matrix location as key:  \n", covariance_matrix)

print()
print()

print("the covariance matrix with np array:  \n", np.matrix(covariance_array))

print()
print()

print("shape of the covariance matrix: ", np.shape(covariance_array))

print()
print()

# for eigen vector and eigen values
# eigenvalues and vector calculation
calc_eigen =  np.array(covariance_array)
eigen_value, eigen_vector = eig(calc_eigen)

print()
print('E-value: \n', eigen_value)

print()

print('E-vector: \n', eigen_vector)
print('shape of E vector', np.shape(eigen_vector))

print()
print('---------sorting data-------------')

sort_eigenvalues = np.sort(eigen_value)
sort_eigenvalues = sort_eigenvalues[::-1]
sum_of_eigenvalues =  np.sum(sort_eigenvalues)
print(sort_eigenvalues)
print("sum : ", sum_of_eigenvalues)

print()
print("-----------assigning pca values-------------")
pca = {}
cumulative_sum_of_eigenvalues = np.cumsum(sort_eigenvalues)
print(sort_eigenvalues)
print(cumulative_sum_of_eigenvalues)
print()

for i in range(len(sort_eigenvalues)):
    pca[(sort_eigenvalues[i]/sum_of_eigenvalues)*100] = sort_eigenvalues[i]

print("overall pca calculation:\n", pca)

print()
# r = np.argmax(cumulative_sum_of_eigenvalues >= 0.9 * np.sum(sort_eigenvalues)) + 1
# print(cumulative_sum_of_eigenvalues >= 0.9 * np.sum(sort_eigenvalues))

pca2 ={}
for i in range(len(cumulative_sum_of_eigenvalues)):
    pca2[(cumulative_sum_of_eigenvalues[i]/sum_of_eigenvalues)*100] = cumulative_sum_of_eigenvalues[i]
print("overall pca calculation after cumulativesum:\n", pca2)
print()

def count_pca(pca, percentage_of_match):
    count_of_pca = 0
    numberof_values = 0
    count = 0
    for i in pca.keys():
        if i >= percentage_of_match:
            # print("the key with minimum PC values required", i)
            count_of_pca = i
            numberof_values = count+1
            break
        count = count+1
    return (count_of_pca, numberof_values)


print("count of pca above : 90 ", count_pca(pca2, 90)[1]," corresponding percentage", count_pca(pca2, 90)[0])
print()
print("count of pca above : 60 ", count_pca(pca2, 60)[1]," corresponding percentage", count_pca(pca2, 60)[0])


plotter_data = np.dot(df_centered.to_numpy(),eigen_vector)
plotter_data_e = np.dot(df_centered.to_numpy(),eigen_vector)[:2,:]


print(plotter_data_e[0])
print(plotter_data_e[1])
# print(plotter_data_e[:,2])
numofpoints = len(plotter_data[0])
plt.plot(range(numofpoints), plotter_data[0], label = 'PC1')
plt.plot(range(numofpoints), plotter_data[1], label = 'PC2')
plt.xlabel('Dimension')
plt.ylabel('Magnitude')
plt.legend()
plt.show()

plotter_data_s = np.dot(df_centered.to_numpy(),eigen_vector[:,:2])
plt.scatter(plotter_data_s[:,0], plotter_data_s[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(df.to_numpy())

xpca = pca.transform(df.to_numpy())
print("-------------printing skylearn pca values ------------------")
# print(pca.explained_variance_ratio_)
print(xpca)

plt.plot(plotter_data[0], label = "calculated by me")
plt.plot(xpca[0], label="sklearn")
plt.legend()
plt.show()


plt.plot(cumulative_sum_of_eigenvalues)
plt.show()


f = open('components.txt', 'w')
for vals in plotter_data_e[0]:
    f.write("%s" %vals)
    f.write(",")
for vals in plotter_data_e[1]:
    f.write("%s" %vals)
    f.write(",")

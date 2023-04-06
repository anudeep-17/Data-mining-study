import numpy as np

# Scoring for classifiers
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import pandas as pd

data = pd.read_csv("cancer-data-train.csv", header = None)
print(data.iloc[:,:-1])
X = data.iloc[:,:-1]
X = X.to_numpy()
X = StandardScaler().fit_transform(X)
print(X)

data[30] = data[30].replace("M", 1)
data[30] = data[30].replace("B", 0)
data[30] = data[30].apply(np.int32)
y = data[30].values
print(y)

k = 10

kf = KFold(n_splits=k)


print('-------------c: 1--------------------')


f1_data1 = []
f1_data2 = []
f1_data3 = []
f1_data4 = []
f1_data5 = []

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],y[train_index], y[test_index]
    classifier = SVC(kernel="linear", C=1)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    f1_data1.append(f1_score(y_test, y_predict,average='weighted'))
    print()
    classifier = SVC(kernel="linear", C=10)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    f1_data2.append(f1_score(y_test, y_predict, average='weighted'))
    print()
    classifier = SVC(kernel="linear", C=100)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    f1_data3.append(f1_score(y_test, y_predict, average='weighted'))
    print()
    classifier = SVC(kernel="linear", C=0.1)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    f1_data4.append(f1_score(y_test, y_predict, average='weighted'))
    print()

    classifier = SVC(kernel="linear", C=0.01)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    f1_data5.append(f1_score(y_test, y_predict, average='weighted'))


print("-------------c = 0.01---------------------------")
print(f1_data1)
print("-------------c = 0.01---------------------------")
print(f1_data2)
print("-------------c = 0.01---------------------------")
print(f1_data3)
print("-------------c = 0.01---------------------------")
print(f1_data4)
print("-------------c = 0.01---------------------------")
print(f1_data5)


def Average(lst):
    return sum(lst) / len(lst)


#-------------plotting ----------------------------
X =["0.01","0.1","1","10","100"]
f1_data = [Average(f1_data5),Average(f1_data4),Average(f1_data1), Average(f1_data2), Average(f1_data3)]
print("------------averaged f1 score------------------------")
print(f1_data)
plt.plot(X,f1_data)
plt.tight_layout()
plt.show()
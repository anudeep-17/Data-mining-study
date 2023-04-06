import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import numpy as np

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


f1_data_IG = []
f1_data_gini = []

f1_data1_gini = []
f1_data2_gini = []
f1_data3_gini = []
f1_data4_gini = []


f1_data1_IG = []
f1_data2_IG = []
f1_data3_IG = []
f1_data4_IG = []



for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],y[train_index], y[test_index]

    print("---------------------K = 2------------------")
    DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=2)
    DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=2)
    DT_gini.fit(X_train, y_train)
    DT_IG.fit(X_train, y_train)
    y_predict_gini = DT_gini.predict(X_test)
    y_predict_IG = DT_IG.predict(X_test)
    f1_data1_gini.append(f1_score(y_test, y_predict_gini,average='weighted'))
    f1_data1_IG.append(f1_score(y_test, y_predict_IG,average='weighted'))
    print()
    print("---------------------K = 5------------------")
    DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=5)
    DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=5)
    DT_gini.fit(X_train, y_train)
    DT_IG.fit(X_train, y_train)
    y_predict_gini = DT_gini.predict(X_test)
    y_predict_IG = DT_IG.predict(X_test)
    f1_data2_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
    f1_data2_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))
    print()
    print("---------------------K = 10------------------")
    DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=10)
    DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=10)
    DT_gini.fit(X_train, y_train)
    DT_IG.fit(X_train, y_train)
    y_predict_gini = DT_gini.predict(X_test)
    y_predict_IG = DT_IG.predict(X_test)
    f1_data3_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
    f1_data3_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))
    print()
    print("---------------------K = 20------------------")
    DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=20)
    DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=20)
    DT_gini.fit(X_train, y_train)
    DT_IG.fit(X_train, y_train)
    y_predict_gini = DT_gini.predict(X_test)
    y_predict_IG = DT_IG.predict(X_test)
    f1_data4_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
    f1_data4_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))
    print()

def Average(lst):
    return sum(lst) / len(lst)


#-------------plotting ----------------------------
X =["2","5","10","20"]
f1_data_IG = [Average(f1_data1_IG),Average(f1_data2_IG),Average(f1_data3_IG), Average(f1_data4_IG)]
f1_data_gini = [Average(f1_data1_gini),Average(f1_data2_gini),Average(f1_data3_gini), Average(f1_data4_gini)]

print("------------averaged f1 score------------------------")
print(f1_data_gini)
print(f1_data_IG)
plt.plot(X,f1_data_gini, label = "GINI")
plt.plot(X,f1_data_IG, label = "IG")
plt.legend()
plt.tight_layout()
plt.show()
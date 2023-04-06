import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random
import numpy as np

class decisiontree:
    f_data_IG = []
    f_data_gini = []

    f1_data1_gini = []
    f1_data2_gini = []
    f1_data3_gini = []
    f1_data4_gini = []

    f1_data1_IG = []
    f1_data2_IG = []
    f1_data3_IG = []
    f1_data4_IG = []
    k = 10
    kf = KFold(n_splits=k)

    X = []
    y = []
    test_X = []
    test_y = []

    def __init__(self, dataset):
        print("--------------decision tree working-------------")
        print()
        decisiontree.X = self.datasplitter(dataset)[0]
        decisiontree.y = self.datasplitter(dataset)[1]
        self.calc(decisiontree.X, decisiontree.y)
        self.plotter()

    def datasplitter(self, dataset):
        data = pd.read_csv(dataset, header=None)
        print(data.iloc[:, :-1])
        X = data.iloc[:, :-1]
        X = X.to_numpy()
        X = StandardScaler().fit_transform(X)
        print(X)

        data[30] = data[30].replace("M", 1)
        data[30] = data[30].replace("B", 0)
        data[30] = data[30].apply(np.int32)
        y = data[30].values
        print(y)
        return (X,y)
    def calc(self, X, y, type = 'Train', testX = test_X, testy = test_y):
        if (type == "Train"):
            print("----------------testing on split data for this ------------")
            print()
        else:
            print("------------------test data passed-----------------")
            print()

        for train_index, test_index in decisiontree.kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=2, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=2, random_state=42)

            if (type == "Train"):
                DT_gini.fit(X_train, y_train)
                DT_IG.fit(X_train, y_train)
                y_predict_gini = DT_gini.predict(X_test)
                y_predict_IG = DT_IG.predict(X_test)
            else:
                DT_gini.fit(X_train + X_test, y_train + y_test)
                DT_IG.fit(X_train + X_test, y_train + y_test)
                y_predict_gini = DT_gini.predict(testX)
                y_predict_IG = DT_IG.predict(testX)
                y_test = testy

            decisiontree.f1_data1_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data1_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=5, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=5, random_state=42)

            if (type == "Train"):
                DT_gini.fit(X_train, y_train)
                DT_IG.fit(X_train, y_train)
                y_predict_gini = DT_gini.predict(X_test)
                y_predict_IG = DT_IG.predict(X_test)
            else:
                DT_gini.fit(X_train + X_test, y_train + y_test)
                DT_IG.fit(X_train + X_test, y_train + y_test)
                y_predict_gini = DT_gini.predict(testX)
                y_predict_IG = DT_IG.predict(testX)
                y_test = testy

            decisiontree.f1_data2_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data2_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=10, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=10, random_state=42)

            if (type == "Train"):
                DT_gini.fit(X_train, y_train)
                DT_IG.fit(X_train, y_train)
                y_predict_gini = DT_gini.predict(X_test)
                y_predict_IG = DT_IG.predict(X_test)
            else:
                DT_gini.fit(X_train + X_test, y_train + y_test)
                DT_IG.fit(X_train + X_test, y_train + y_test)
                y_predict_gini = DT_gini.predict(testX)
                y_predict_IG = DT_IG.predict(testX)
                y_test = testy

            decisiontree.f1_data3_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data3_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=20, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=20, random_state=42)

            if (type == "Train"):
                DT_gini.fit(X_train, y_train)
                DT_IG.fit(X_train, y_train)
                y_predict_gini = DT_gini.predict(X_test)
                y_predict_IG = DT_IG.predict(X_test)
            else:
                DT_gini.fit(X_train + X_test, y_train + y_test)
                DT_IG.fit(X_train + X_test, y_train + y_test)
                y_predict_gini = DT_gini.predict(testX)
                y_predict_IG = DT_IG.predict(testX)
                y_test = testy

            decisiontree.f1_data4_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data4_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

        print("---------------------K = 2------------------")
        print("f1 score for gini",  decisiontree.f1_data1_gini)
        print("f1 score for IG",  decisiontree.f1_data1_IG)
        print()
        print("---------------------K = 5------------------")
        print("f1 score for gini",  decisiontree.f1_data2_gini)
        print("f1 score for IG",  decisiontree.f1_data2_IG)
        print()
        print("---------------------K = 10------------------")
        print("f1 score for gini",  decisiontree.f1_data3_gini)
        print("f1 score for IG",  decisiontree.f1_data3_IG)
        print()
        print("---------------------K = 20------------------")
        print("f1 score for gini",  decisiontree.f1_data4_gini)
        print("f1 score for IG",  decisiontree.f1_data4_IG)
        print()

    def Average(self, lst):
        return sum(lst) / len(lst)

    def plotter(self):
        # -------------plotting ----------------------------
        X = ["2", "5", "10", "20"]
        f_data_IG = [self.Average(decisiontree.f1_data1_IG), self.Average(decisiontree.f1_data2_IG), self.Average(decisiontree.f1_data3_IG), self.Average(decisiontree.f1_data4_IG)]
        f_data_gini = [self.Average(decisiontree.f1_data1_gini), self.Average(decisiontree.f1_data2_gini), self.Average(decisiontree.f1_data3_gini), self.Average(decisiontree.f1_data4_gini)]

        print("------------averaged f1 score------------------------")
        print("f1 scores of gini: ", f_data_gini)
        print("f1 scores of IG: ", f_data_IG)
        plt.plot(X, f_data_gini, label="GINI")
        plt.plot(X, f_data_IG, label="IG")
        plt.legend()
        plt.tight_layout()
        plt.show()
        decisiontree.f_data_gini = f_data_gini
        decisiontree.f_data_IG = f_data_IG

    def testreport(self, testdataset):
        print("------------testing of the trained model started--------")
        decisiontree.test_X = self.datasplitter(testdataset)[0]
        decisiontree.test_y = self.datasplitter(testdataset)[1]
        self.calc(decisiontree.X, decisiontree.y, type="test", testX =  decisiontree.test_X,testy=decisiontree.test_y)
        self.plotter()


# decisiontreetest = decisiontree("cancer-data-train.csv")
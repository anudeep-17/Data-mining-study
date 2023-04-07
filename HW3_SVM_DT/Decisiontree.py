import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score , confusion_matrix, f1_score, recall_score
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

    confusion_matrix_gini = []
    precision_score_gini = 0
    recall_score_gini = 0
    F1_score_gini = 0

    confusion_matrix_IG = []
    precision_score_IG = 0
    recall_score_IG = 0
    F1_score_IG = 0

    def __init__(self, dataset, testdataset):
        print("--------------decision tree working-------------")
        # print()
        decisiontree.X = self.datasplitter(dataset)[0]
        decisiontree.y = self.datasplitter(dataset)[1]
        self.calc()
        self.plotter()

        decisiontree.confusion_matrix_gini = self.testreport(testdataset)[0][0]
        decisiontree.precision_score_gini = self.testreport(testdataset)[0][1]
        decisiontree.recall_score_gini = self.testreport(testdataset)[0][2]
        decisiontree.F1_score_gini = self.testreport(testdataset)[0][3]

        decisiontree.confusion_matrix_IG = self.testreport(testdataset)[1][0]
        decisiontree.precision_score_IG = self.testreport(testdataset)[1][1]
        decisiontree.recall_score_IG = self.testreport(testdataset)[1][2]
        decisiontree.F1_score_IG = self.testreport(testdataset)[1][3]


    def datasplitter(self, dataset):
        data = pd.read_csv(dataset, header=None)
        # print(data.iloc[:, :-1])
        X = data.iloc[:, :-1]
        X = X.to_numpy()
        X = StandardScaler().fit_transform(X)
        # print(X)

        data[30] = data[30].replace("M", 1)
        data[30] = data[30].replace("B", 0)
        data[30] = data[30].apply(np.int32)
        y = data[30].values
        # print(y)
        return (X,y)

    def calc(self):

        for train_index, test_index in decisiontree.kf.split(decisiontree.X):
            X_train, X_test, y_train, y_test = decisiontree.X[train_index], decisiontree.X[test_index], decisiontree.y[train_index], decisiontree.y[test_index]

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=2, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=2, random_state=42)
            DT_gini.fit(X_train, y_train)
            DT_IG.fit(X_train, y_train)
            y_predict_gini = DT_gini.predict(X_test)
            y_predict_IG = DT_IG.predict(X_test)
            decisiontree.f1_data1_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data1_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=5, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=5, random_state=42)
            DT_gini.fit(X_train, y_train)
            DT_IG.fit(X_train, y_train)
            y_predict_gini = DT_gini.predict(X_test)
            y_predict_IG = DT_IG.predict(X_test)
            decisiontree.f1_data2_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data2_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=10, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=10, random_state=42)
            DT_gini.fit(X_train, y_train)
            DT_IG.fit(X_train, y_train)
            y_predict_gini = DT_gini.predict(X_test)
            y_predict_IG = DT_IG.predict(X_test)
            decisiontree.f1_data3_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data3_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

            DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=20, random_state=42)
            DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=20, random_state=42)
            DT_gini.fit(X_train, y_train)
            DT_IG.fit(X_train, y_train)
            y_predict_gini = DT_gini.predict(X_test)
            y_predict_IG = DT_IG.predict(X_test)
            decisiontree.f1_data4_gini.append(f1_score(y_test, y_predict_gini, average='weighted'))
            decisiontree.f1_data4_IG.append(f1_score(y_test, y_predict_IG, average='weighted'))

        # print("---------------------K = 2------------------")
        # print("f1 score for gini",  decisiontree.f1_data1_gini)
        # print("f1 score for IG",  decisiontree.f1_data1_IG)
        # print()
        # print("---------------------K = 5------------------")
        # print("f1 score for gini",  decisiontree.f1_data2_gini)
        # print("f1 score for IG",  decisiontree.f1_data2_IG)
        # print()
        # print("---------------------K = 10------------------")
        # print("f1 score for gini",  decisiontree.f1_data3_gini)
        # print("f1 score for IG",  decisiontree.f1_data3_IG)
        # print()
        # print("---------------------K = 20------------------")
        # print("f1 score for gini",  decisiontree.f1_data4_gini)
        # print("f1 score for IG",  decisiontree.f1_data4_IG)
        # print()

    def Average(self, lst):
        return sum(lst) / len(lst)

    def plotter(self):
        # -------------plotting ----------------------------
        X = ["2", "5", "10", "20"]
        decisiontree.f_data_IG = [self.Average(decisiontree.f1_data1_IG), self.Average(decisiontree.f1_data2_IG), self.Average(decisiontree.f1_data3_IG), self.Average(decisiontree.f1_data4_IG)]
        decisiontree.f_data_gini = [self.Average(decisiontree.f1_data1_gini), self.Average(decisiontree.f1_data2_gini), self.Average(decisiontree.f1_data3_gini), self.Average(decisiontree.f1_data4_gini)]

        print("------------averaged f1 score------------------------")
        print("f1 scores of gini: ", decisiontree.f_data_gini)
        print("f1 scores of IG: ", decisiontree.f_data_IG)
        plt.plot(X, decisiontree.f_data_gini, label="GINI")
        plt.plot(X, decisiontree.f_data_IG, label="IG")
        plt.xlabel("K values")
        plt.ylabel("f1 scores")
        plt.title("DT")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def testreport(self, testdataset):
        # print("------------testing of the trained model started--------")
        decisiontree.test_X = self.datasplitter(testdataset)[0]
        decisiontree.test_y = self.datasplitter(testdataset)[1]

        X = [2,5,10,20]

        Best_criterion_gini = decisiontree.f_data_gini.index(max(decisiontree.f_data_gini))
        Best_criterion_IG = decisiontree.f_data_IG.index(max(decisiontree.f_data_IG))

        # print(Best_criterion_IG, Best_criterion_gini)

        DT_gini = DecisionTreeClassifier(criterion="gini", splitter="best", max_leaf_nodes=X[Best_criterion_gini], random_state=42)
        DT_IG = DecisionTreeClassifier(criterion="entropy", splitter="best", max_leaf_nodes=X[Best_criterion_IG], random_state=42)

        DT_gini.fit(decisiontree.X,decisiontree.y)
        DT_IG.fit(decisiontree.X, decisiontree.y)

        y_predict_gini = DT_gini.predict(decisiontree.test_X)
        y_predict_IG = DT_IG.predict(decisiontree.test_X)

        return (
                (confusion_matrix(decisiontree.test_y, y_predict_gini)
                 ,average_precision_score(decisiontree.test_y, y_predict_gini),
                 recall_score(decisiontree.test_y, y_predict_gini, average="weighted")
                 ,f1_score(decisiontree.test_y, y_predict_gini, average='weighted')
                 )
                ,
                (confusion_matrix(decisiontree.test_y, y_predict_IG),
                 average_precision_score(decisiontree.test_y, y_predict_IG),
                 recall_score(decisiontree.test_y, y_predict_IG, average="weighted") ,
                 f1_score(decisiontree.test_y, y_predict_IG, average='weighted')
                 )
                )

# decisiontreetest = decisiontree("cancer-data-train.csv")
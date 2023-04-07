import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Scoring for classifiers
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import pandas as pd


class RFC:
    test_X = []
    test_y = []
    X = []
    y = []

    f1_data = []

    k = 10
    kf = KFold(n_splits=k)
    confusion_matrix = []
    precision_score = 0
    recall_score = 0
    F1_score = 0


    def __init__(self, dataset, testdataset):
        print("----------------Random Forest Classifier-------------------------")
        # print()
        RFC.X = self.datasplitter(dataset)[0]
        RFC.y = self.datasplitter(dataset)[1]
        self.calc()
        RFC.confusion_matrix = self.testreport(testdataset)[0]
        RFC.precision_score = self.testreport(testdataset)[1]
        RFC.recall_score = self.testreport(testdataset)[2]
        RFC.F1_score = self.testreport(testdataset)[3]

    def datasplitter(self, dataset):
        data = pd.read_csv(dataset, header=None)
        # print(data.iloc[:, :-1])
        X = data.iloc[:, :-1]
        X =  X.to_numpy()
        X = StandardScaler().fit_transform(X)
        # print(RFC.X)

        data[30] = data[30].replace("M", 1)
        data[30] = data[30].replace("B", 0)
        data[30] = data[30].apply(np.int32)
        y = data[30].values
        # print(RFC.y)
        return (X,y)

    def calc(self):
        for train_index, test_index in RFC.kf.split(RFC.X):
            X_train, X_test, y_train, y_test = RFC.X[train_index], RFC.X[test_index], RFC.y[train_index], RFC.y[test_index]
            classifier =  RandomForestClassifier()
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            RFC.f1_data.append(f1_score(y_test, y_predict, average='weighted'))
            # print()
        # print()
        print("f1 scores of RFC: ", RFC.f1_data)

    def testreport(self, testdataset):
        # print("------------testing of the trained model started--------")
        RFC.test_X = self.datasplitter(testdataset)[0]
        RFC.test_y = self.datasplitter(testdataset)[1]
        classifier = RandomForestClassifier()
        classifier.fit(RFC.X, RFC.y)
        y_predict = classifier.predict(RFC.test_X)
        return (confusion_matrix(RFC.test_y, y_predict),
                average_precision_score(RFC.test_y, y_predict),
                recall_score(RFC.test_y, y_predict, average="weighted"),
                f1_score(RFC.test_y, y_predict, average='weighted'))


import numpy as np

# Scoring for classifiers
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import average_precision_score

from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import pandas as pd


class LDA:
    test_X = []
    test_y = []
    X = []
    y = []

    f1_data = []

    k = 10
    kf = KFold(n_splits=k)

    def __init__(self, dataset):
        print("---------------------LDA----------------------------")
        print()
        LDA.X = self.datasplitter(dataset)[0]
        LDA.y = self.datasplitter(dataset)[1]
        self.calc()

    def datasplitter(self, dataset):
        data = pd.read_csv("cancer-data-train.csv", header=None)
        print(data.iloc[:, :-1])
        LDA.X = data.iloc[:, :-1]
        LDA.X = LDA.X.to_numpy()
        LDA.X = StandardScaler().fit_transform(LDA.X)
        print(LDA.X)

        data[30] = data[30].replace("M", 1)
        data[30] = data[30].replace("B", 0)
        data[30] = data[30].apply(np.int32)
        LDA.y = data[30].values
        print(LDA.y)
        return (LDA.X, LDA.y)

    def calc(self):
        for train_index, test_index in LDA.kf.split(LDA.X):
            X_train, X_test, y_train, y_test = LDA.X[train_index], LDA.X[test_index], LDA.y[train_index], LDA.y[
                test_index]
            classifier =LinearDiscriminantAnalysis()
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            LDA.f1_data.append(f1_score(y_test, y_predict, average='weighted'))
            # print()
        print()
        print("f1 scores of LDA: ", LDA.f1_data)


    def testreport(self, testdataset):
        print("------------testing of the trained model started--------")
        LDA.test_X = self.datasplitter(testdataset)[0]
        LDA.test_y = self.datasplitter(testdataset)[1]
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(LDA.X, LDA.y)
        y_predict = classifier.predict(LDA.test_X)
        return (precision_score(LDA.test_y, y_predict), recall_score(LDA.test_y, y_predict),
                f1_score(LDA.test_y, y_predict, average='weighted'))

# LDAtest = LDA("cancer-data-train.csv")
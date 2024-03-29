import numpy as np

# Scoring for classifiers
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
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
    confusion_matrix = []
    precision_score = 0
    recall_score = 0
    F1_score = 0

    def __init__(self, dataset, testdataset):
        print("---------------------LDA----------------------------")
        # print()
        LDA.X = self.datasplitter(dataset)[0]
        LDA.y = self.datasplitter(dataset)[1]
        self.calc()
        LDA.confusion_matrix = self.testreport(testdataset)[0]
        LDA.precision_score = self.testreport(testdataset)[1]
        LDA.recall_score = self.testreport(testdataset)[2]
        LDA.F1_score = self.testreport(testdataset)[3]

    def datasplitter(self, dataset):
        data = pd.read_csv(dataset, header=None)
        # print(data.iloc[:, :-1])
        X = data.iloc[:, :-1]
        X = X.to_numpy()
        X = StandardScaler().fit_transform(X)
        # print(LDA.X)

        data[30] = data[30].replace("M", 1)
        data[30] = data[30].replace("B", 0)
        data[30] = data[30].apply(np.int32)
        y = data[30].values
        # print(LDA.y)
        return (X, y)

    def calc(self):
        for train_index, test_index in LDA.kf.split(LDA.X):
            X_train, X_test, y_train, y_test = LDA.X[train_index], LDA.X[test_index], LDA.y[train_index], LDA.y[
                test_index]
            classifier = LinearDiscriminantAnalysis()
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            LDA.f1_data.append(f1_score(y_test, y_predict, average='weighted'))
            # print()
        print()
        print("f1 scores of LDA: ", LDA.f1_data)


    def testreport(self, testdataset):
        # print("------------testing of the trained model started--------")
        LDA.test_X = self.datasplitter(testdataset)[0]
        LDA.test_y = self.datasplitter(testdataset)[1]
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(LDA.X, LDA.y)
        y_predict = classifier.predict(LDA.test_X)
        return (confusion_matrix(LDA.test_y, y_predict), average_precision_score(LDA.test_y, y_predict), recall_score(LDA.test_y, y_predict, average = "weighted"),
                f1_score(LDA.test_y, y_predict, average='weighted'))

# LDAtest = LDA("cancer-data-train.csv")
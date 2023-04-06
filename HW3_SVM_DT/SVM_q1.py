import numpy as np

# Scoring for classifiers
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# SVM: linear and a kernel-SVM (you can read more about it in the SVM chapter)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import pandas as pd
class SVM:
    f1_data1 = []
    f1_data2 = []
    f1_data3 = []
    f1_data4 = []
    f1_data5 = []
    test_X = []
    test_y = []
    X = []
    y = []
    f1_data = []

    k = 10
    kf = KFold(n_splits=k)
    confusion_matrix = []
    precision_score=0
    recall_score = 0
    F1_score = 0

    def __init__(self, dataset, testdataset):
        print("-------------------SVM------------------")
        SVM.X = self.datasplitter(dataset)[0]
        SVM.y = self.datasplitter(dataset)[1]
        self.calc()
        self.plotter()
        SVM.confusion_matrix = self.testreport(testdataset)[0]
        SVM.precision_score = self.testreport(testdataset)[1]
        SVM.recall_score = self.testreport(testdataset)[2]
        SVM.F1_score = self.testreport(testdataset)[3]

    def datasplitter(self, dataset):
        data = pd.read_csv(dataset, header=None)
        # print(data.iloc[:, :-1])
        SVM.X = data.iloc[:, :-1]
        SVM.X = SVM.X.to_numpy()
        SVM.X = StandardScaler().fit_transform(SVM.X)
        # print(SVM.X)

        data[30] = data[30].replace("M", 1)
        data[30] = data[30].replace("B", 0)
        data[30] = data[30].apply(np.int32)
        SVM.y = data[30].values
        # print(SVM.y)
        return (SVM.X, SVM.y)

    def calc(self):
        # print('-------------c: 1--------------------')

        for train_index, test_index in SVM.kf.split(SVM.X):
            X_train, X_test, y_train, y_test = SVM.X[train_index], SVM.X[test_index], SVM.y[train_index], SVM.y[test_index]
            classifier = SVC(kernel="linear", C=1)
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            SVM.f1_data1.append(f1_score(y_test, y_predict, average='weighted'))
            # print()
            classifier = SVC(kernel="linear", C=10)
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            SVM.f1_data2.append(f1_score(y_test, y_predict, average='weighted'))
            # print()
            classifier = SVC(kernel="linear", C=100)
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            SVM.f1_data3.append(f1_score(y_test, y_predict, average='weighted'))
            # print()
            classifier = SVC(kernel="linear", C=0.1)
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            SVM.f1_data4.append(f1_score(y_test, y_predict, average='weighted'))
            # print()

            classifier = SVC(kernel="linear", C=0.01)
            classifier.fit(X_train, y_train)
            y_predict = classifier.predict(X_test)
            SVM.f1_data5.append(f1_score(y_test, y_predict, average='weighted'))

        # print("-------------c = 1---------------------------")
        # print(SVM.f1_data1)
        # print("-------------c = 10---------------------------")
        # print(SVM.f1_data2)
        # print("-------------c = 100---------------------------")
        # print(SVM.f1_data3)
        # print("-------------c = 0.1---------------------------")
        # print(SVM.f1_data4)
        # print("-------------c = 0.01---------------------------")
        # print(SVM.f1_data5)


    def Average(self,lst):
        return sum(lst) / len(lst)


    def plotter(self):
        # -------------plotting ----------------------------
        X = ["0.01", "0.1", "1", "10", "100"]
        SVM.f1_data = [self.Average(SVM.f1_data5), self.Average(SVM.f1_data4), self.Average(SVM.f1_data1), self.Average(SVM.f1_data2), self.Average(SVM.f1_data3)]
        print("------------averaged f1 score------------------------")
        print(SVM.f1_data)
        plt.plot(X, SVM.f1_data)
        plt.tight_layout()
        plt.show()

    def testreport(self, testdataset):
        print("------------testing of the trained model started--------")
        SVM.test_X = self.datasplitter(testdataset)[0]
        SVM.test_y = self.datasplitter(testdataset)[1]
        X = [0.01,0.1,1,10,100]
        Best_Cval= SVM.f1_data.index(max(SVM.f1_data))
        classifier = SVC(kernel="linear", C=X[Best_Cval])
        print(classifier)
        classifier.fit(SVM.X, SVM.y)
        y_predict = classifier.predict(SVM.test_X)
        return (confusion_matrix(SVM.test_y, y_predict),average_precision_score(SVM.test_y, y_predict), recall_score(SVM.test_y, y_predict, average="weighted") ,f1_score(SVM.test_y, y_predict, average='weighted'))

import matplotlib.pyplot as plt
from SVM_q1 import SVM
from LDA import LDA
from Decisiontree import decisiontree
from RandomForestClassifier import RFC

precesion_allmodels = []
fscore_allmodels= []
recall_allmodels=[]
confusion_matrix_allmodels = []

SVM = SVM("cancer-data-train.csv", "cancer-data-test.csv")
precesion_allmodels.append(SVM.precision_score)
recall_allmodels.append(SVM.recall_score)
fscore_allmodels.append(SVM.F1_score)
confusion_matrix_allmodels.append(SVM.confusion_matrix)

LDA = LDA("cancer-data-train.csv", "cancer-data-test.csv")
precesion_allmodels.append(LDA.precision_score)
recall_allmodels.append(LDA.recall_score)
fscore_allmodels.append(LDA.F1_score)
confusion_matrix_allmodels.append(LDA.confusion_matrix)

print()
DT = decisiontree("cancer-data-train.csv", "cancer-data-test.csv")
precesion_allmodels.append(DT.precision_score_gini)
recall_allmodels.append(DT.recall_score_gini)
fscore_allmodels.append(DT.F1_score_gini)
confusion_matrix_allmodels.append(DT.confusion_matrix_gini)

precesion_allmodels.append(DT.precision_score_IG)
recall_allmodels.append(DT.recall_score_IG)
fscore_allmodels.append(DT.F1_score_IG)
confusion_matrix_allmodels.append(DT.confusion_matrix_IG)

print()
RFC = RFC("cancer-data-train.csv", "cancer-data-test.csv")
precesion_allmodels.append(RFC.precision_score)
recall_allmodels.append(RFC.recall_score)
fscore_allmodels.append(RFC.F1_score)
confusion_matrix_allmodels.append(RFC.confusion_matrix)

print()
print()

print("============ REPORT ==================")
Xaxis = ["SVM", "LDA", "DT-GINI", "DT-IG", "RFC"]

print("classifiers order : ", Xaxis)
print("precesion scores: ", precesion_allmodels)
print("recall scores: ",recall_allmodels)
print("f1 score scores: ", fscore_allmodels)

print("SVM: " ,confusion_matrix_allmodels[0])
print("LDA: " ,confusion_matrix_allmodels[1])
print("DT-GINI: " ,confusion_matrix_allmodels[2])
print("DT-IG: " ,confusion_matrix_allmodels[3])
print("RFC: " ,confusion_matrix_allmodels[4])
# plotting ------------------------------------------

plt.bar(Xaxis, precesion_allmodels)
plt.xlabel("Classifier")
plt.ylabel("precession values")
plt.title("bar plot - precession")
plt.show()


plt.bar(Xaxis, recall_allmodels)
plt.xlabel("Classifier")
plt.ylabel("recall values")
plt.title("bar plot - recall")
plt.show()


plt.bar(Xaxis, fscore_allmodels)
plt.xlabel("Classifier")
plt.ylabel("fscore values")
plt.title("bar plot - fscore")
plt.show()
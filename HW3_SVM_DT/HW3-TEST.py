import matplotlib.pyplot as plt
from SVM_q1 import SVM
from LDA import LDA
from Decisiontree import decisiontree
from RandomForestClassifier import RFC

precesion_allmodels = []
fscore_allmodels= []
recall_allmodels=[]

SVM = SVM("cancer-data-train.csv", "cancer-data-test.csv")
precesion_allmodels.append(SVM.precision_score)
recall_allmodels.append(SVM.recall_score)
fscore_allmodels.append(SVM.F1_score)

# LDA = LDA("cancer-data-train.csv")
# DT = decisiontree("cancer-data-train.csv")
# RFC = RFC("cancer-data-train.csv")
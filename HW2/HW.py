import math

import numpy as np
import pandas as pd
data = pd.read_csv('train.txt', header=None)
X = data.iloc[:, :len(data.columns)-1]
Y = data.iloc[:, len(data.columns)-1:len(data.columns)]
print(X)
def splitter(D, index, value):
    D_y = []
    D_n = []
    for i in range(len(D[index])):
        if (D[index][i] <= value):
            D_y.append(D.iloc[i])
        else:
            D_n.append(D.iloc[i])
    return (D_y, D_n)

def wholeset_entropy(D, index):
    if len(D) == 0:
        return (0, 0, 0)

    D_y = []
    D_n = []
    # print(len(D[index]))
    for i in range(len(D[index])):
        print(D[index][i])
        if (D[10][i] == 1):
            D_y.append(D[index][i])
        else:
            D_n.append(D[index][i])

    if len(D_y) != 0 and len(D_n) != 0:
        whole_set_entropy = -(
        ((len(D_y) / len(D)) * math.log2(len(D_y) / len(D)) + (len(D_n) / len(D)) * math.log2(len(D_n) / len(D))))

    elif len(D_y) == 0 and len(D_n) != 0:
        whole_set_entropy = -((0 + (len(D_n) / len(D)) * math.log2(len(D_n) / len(D))))

    elif len(D_y) != 0 and len(D_n) == 0:
        whole_set_entropy = -((0 + (len(D_y) / len(D)) * math.log2(len(D_y) / len(D))))

    Gini_index = 1 - (math.pow(((len(D_y) / len(D))), 2)) - (math.pow(((len(D_n) / len(D))), 2))

    # print(len(D_n))
    prob_difference = math.fabs(((len(D_y) / len(D))) - ((len(D_n) / len(D))))
    # print(prob_difference)

    return (whole_set_entropy, Gini_index, prob_difference)


#given methods
def IG(D, index, value):
    """Compute the Information Gain of a split on attribute index at value
    for dataset D.
    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at
    Returns:
        The value of the Information Gain for the given split
    """
    D_y = splitter(D[0],index, value)[0]
    D_n = splitter(D[0],index, value)[1]
    # print(len(D_y), len(D_n))
    print("after split: D_y -- \n", pd.DataFrame(splitter(D[0],index, value)[0]))
    print()
    print("after split: D_n -- \n", pd.DataFrame(splitter(D[0],index, value)[1]))


    entropy_divide = (len(D_y)/len(D)) * wholeset_entropy((pd.DataFrame(D_y).reset_index(drop=True), D[1]),index)[0] + (len(D_n)/len(D))  * wholeset_entropy((pd.DataFrame(D_y).reset_index(drop=True),D[1]), index)[0]
    IG = wholeset_entropy((D[0], D[1]), index)[0] - entropy_divide
    print(entropy_divide)
    print("found information gain: ", IG)
    return IG

#testing
IG((X,Y), 0, 1)
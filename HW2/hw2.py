import math

import numpy as np
import pandas as pd

# data = np.genfromtxt('t.txt', dtype=int, encoding=None,delimiter=",")
data = pd.read_csv('t.txt', header = None)

#helper methods:
def splitter(D, index, value):
    D_y = []
    D_n = []
    for i in range(len(D[index])):
        if (D[index][i] >= value):
            D_y.append(D.iloc[i])
        else:
            D_n.append(D.iloc[i])
    return (D_y, D_n)

def wholeset_entropy(D, index):
    D_y = []
    D_n = []
    for i in range(len(D[index])):
        if (D[10][i] >= 1):
            D_y.append(D[index][i])
        else:
            D_n.append(D[index][i])

    whole_set_entropy = -(((len(D_y) / len(D)) * math.log2(len(D_y) / len(D)) + (len(D_n) / len(D)) * math.log2(len(D_n) / len(D))))
    Gini_index = 1 - (math.pow(((len(D_y) / len(D))), 2)) - (math.pow(((len(D_n) / len(D))), 2))
    prob_difference = math.fabs(((len(D_y) / len(D))) - ((len(D_n) / len(D))))
    return (whole_set_entropy,Gini_index, prob_difference)

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
    D_y = splitter(D,index, value)[0]
    D_n = splitter(D,index, value)[0]
    print("after split: D_y -- \n", pd.DataFrame(splitter(D,index, value)[0]))
    print()
    print("after split: D_n -- \n", pd.DataFrame(splitter(D,index, value)[1]))

    entropy_divide = (len(D_y)/len(D)) * wholeset_entropy(pd.DataFrame(D_y).reset_index(drop=True),index)[0] + (len(D_n)/len(D))  * wholeset_entropy(pd.DataFrame(D_n).reset_index(drop=True),index)[0]
    IG = wholeset_entropy(D, index)[0] - entropy_divide
    print()
    print("found information gain: ", IG)
    return IG

#testing
IG(data, 1, 28)

def G(D, index, value):
    """Compute the Gini index of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Gini index for the given split
    """
    D_y = splitter(D, index, value)[0]
    D_n = splitter(D, index, value)[1]

    gini_index = (len(D_y)/len(D)) * wholeset_entropy(pd.DataFrame(D_y).reset_index(drop=True),index)[1] + (len(D_n)/len(D))  * wholeset_entropy(pd.DataFrame(D_n).reset_index(drop=True),index)[1]
    print("calculated gini index: ", gini_index)
    return gini_index

#testing
G(data, 1, 28)


def CART(D, index, value):
    """Compute the CART measure of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the CART measure for the given split
    """
    D_y = splitter(D,index,value)[0]
    D_n = splitter(D,index,value)[1]
    cart = 2 * (len(D_y)/len(D)) * (len(D_n)/len(D)) * (wholeset_entropy(D,index)[2])
    print("Cart measure: ", cart)
    return cart

#testing
CART(data,1,28)

def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """


# functions are first class objects in python, so let's refer to our desired criterion by a single name


def load(filename):
    """Loads filename as a dataset. Assumes the last column is classes, and
    observations are organized as rows.

    Args:
        filename: file to read

    Returns:
        A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
        where X[i] comes from the i-th row in filename; y is a list or ndarray of
        the classes of the observations, in the same order
    """


def classifyIG(train, test):
    """Builds a single-split decision tree using the Information Gain criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """


def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """


def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """


def main():
    """This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point
    unlike C, Java, etc.
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
    """


if __name__ == "__main__":
    """__name__=="__main__" when the python script is run directly, not when it 
    is imported. When this program is run from the command line (or an IDE), the 
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
    """
    main()
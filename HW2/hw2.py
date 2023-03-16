import math

import numpy as np
import pandas as pd

data = pd.read_csv('train.txt', header=None)

#helper methods:
def info_prompter(train_data):
    """ info_prompter: takes in a loaded data and prompts the user the best splits of each criterion and thw whole given data
        Args: a tuple with whole data, best splits of IG, G, and CART - ((x,y), bestsplit_IG, bestsplit_G, bestsplit_CART)
    """
    print("whole data: \n", pd.concat((train_data[0][0], train_data[0][1]), axis = 1)) #please check the shape of arguments to know why data is called like this.
    print()
    print("---------overall info----------")
    print("best IG at : ", train_data[1])
    print("best GINI at : ", train_data[2])
    print("best CART at : ", train_data[3])
    print()

def splitter(D, index, value):
    """
    splitter method is used to take data and split the given index at the given value and return 2 dataframes where one has values <= given value and the other have greater than given value
    Args:
        D: data on which we are splitting
        index: index of the attribute that is going to be split for the given
        value: the value at which we compare and perform split

    Returns: (D_y, D_n) where D_y has all 'values<=value' and D_n has 'values>value'

    """
    D_y = []
    D_n = []
    for i in range(len(D[index])):
        if (D[index][i] <= value):
            D_y.append(D.iloc[i]) #choosing whole row from data to have clear idea.
        else:
            D_n.append(D.iloc[i]) #choosing whole row from data to have clear idea.
    return (D_y, D_n)

def error_index(data, true_class):
    """
    claculates the error of the classification
     Args:
       data: data that is calssified
       true_class: predicted class of the data to compare the value and count errors.

     Returns: count of classification errors.
    """

    check = float(true_class)
    if data[len(data.columns)-1].value_counts()[check] == len(data):
        return 0
    else:
        return len(data) - data[len(data.columns)-1].value_counts()[check]

def wholeset_entropy(D, index):
    """
    wholeset entropy is the key method that calculates the entropy, Gini value, probability difference for CART and purity of split of a given data
    Args:
        D: whole or partial data set for calculation of entropy, gini value or probability difference to support calculation of IG, G, and cart values.
        index: the index on which the data is split and compare it accordingly.

    Returns: a tuple with entropyvalue, ginivalue, probalbility difference, and purity of the split of data

    """
    if len(D) == 0:
        return (0,0,0)

    D_y = []
    D_n = []
    # print(len(D[index]))
    #splits the data according the its true class values for calculation and comparision.
    for i in range(len(D[index])):
        # print(D[index][i])
        if (D[len(D.columns)-1][i] == 1):
            D_y.append(D[index][i])
        else:
            D_n.append(D[index][i])

    # calculates the entropy of given data.
    if len(D_y) != 0 and len(D_n) != 0:
        whole_set_entropy = -(((len(D_y) / len(D)) * math.log2(len(D_y) / len(D)) + (len(D_n) / len(D)) * math.log2(len(D_n) / len(D))))

    elif len(D_y) == 0 and len(D_n) != 0:
        whole_set_entropy = -((0 + (len(D_n) / len(D)) * math.log2(len(D_n) / len(D))))

    elif len(D_y) != 0 and len(D_n) == 0:
        whole_set_entropy = -((0 + (len(D_y) / len(D)) * math.log2(len(D_y) / len(D))))

    #calculates the gini value of given data.
    Gini_index = 1 - (math.pow(((len(D_y) / len(D))), 2)) - (math.pow(((len(D_n) / len(D))), 2))

    # print(len(D_n))
    # calculates probability difference for cart calculation
    prob_difference = math.fabs(((len(D_y) / len(D))) - ((len(D_n) / len(D))))
    # print(prob_difference)

    #calculates the purity of the split.
    if max(len(D_y)/len(D), len(D_n)/len(D)) == len(D_y)/len(D):
        # print(len(D_y)/len(D),len(D_n)/len(D))
        purityindex = 1
    else:
        purityindex = 0

    return (whole_set_entropy,Gini_index, prob_difference, purityindex)



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
    data = pd.concat((D[0], D[1]), axis=1)
    # print(data)
    D_y = splitter(data,index, value)[0]
    D_n = splitter(data,index, value)[1]

    entropy_divide = (len(D_y)/len(data)) * wholeset_entropy(pd.DataFrame(D_y).reset_index(drop=True),index)[0] + (len(D_n)/len(data))  * wholeset_entropy(pd.DataFrame(D_n).reset_index(drop=True),index)[0]
    IG = wholeset_entropy(data, index)[0] - entropy_divide
    # print(entropy_divide)
    # print("found information gain: ", IG)
    return IG

#testing
# IG(data, 0, 1)

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
    data = pd.concat((D[0], D[1]), axis=1)

    D_y = splitter(data, index, value)[0]
    D_n = splitter(data, index, value)[1]

    if len(D_y) != 0 and len(D_n) != 0:
        gini_index = (len(D_y) / len(data)) * wholeset_entropy(pd.DataFrame(D_y).reset_index(drop=True), index)[1] + (len(D_n) / len(data)) * wholeset_entropy(pd.DataFrame(D_n).reset_index(drop=True), index)[1]

    elif len(D_y) != 0 and len(D_n) == 0:
        gini_index = (len(D_y) / len(data)) * wholeset_entropy(pd.DataFrame(D_y).reset_index(drop=True), index)[1]

    elif len(D_y) == 0 and len(D_n) != 0:
        gini_index = (len(D_n) / len(data)) * wholeset_entropy(pd.DataFrame(D_n).reset_index(drop=True), index)[1]

    # print("calculated gini index: ", gini_index)
    return gini_index

#testing
# G(data, 1, 28)


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

    data = pd.concat((D[0], D[1]), axis=1)

    D_y = splitter(data,index,value)[0]
    D_n = splitter(data,index,value)[1]
    # print(len(D_y), len(D_n))
    cart = 2 * (len(D_y)/len(data)) * (len(D_n)/len(data)) * (wholeset_entropy(pd.DataFrame(D_y).reset_index(drop=True),index)[2] + wholeset_entropy(pd.DataFrame(D_n).reset_index(drop=True),index)[2])
    # print("Cart measure: ", cart)
    return cart

#testing
# CART(data,1,28)

def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion
    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"
    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """

    correspondingsplits = {}

    if(criterion.upper() == "IG"):
        # print('chosen : IG')
        # print()
        for i in D[0].columns:
            uniques = D[0][i].unique()
            for j in uniques:
                correspondingsplits[IG((D[0], D[1]), i, j)] = (i,j)
        # print("largest Information gain found: " , max(correspondingsplits), "with the attribute and value being", correspondingsplits.get(max(correspondingsplits)))
        return correspondingsplits.get(max(correspondingsplits))

    elif (criterion.upper() == "GINI"):
        # print()
        # print('chosen : GINI')
        # print()

        for i in D[0].columns:
            uniques = D[0][i].unique()
            # print(uniques)
            for j in uniques:
                correspondingsplits[G((D[0], D[1]), i, j)] = (i,j)
        # print("smallest GINI found: ", min(correspondingsplits), "with the attribute and value being", correspondingsplits.get(min(correspondingsplits)))
        return correspondingsplits.get(min(correspondingsplits))

    elif (criterion.upper() == "CART"):
    #     print()
    #     print('chosen: CART')
    #     print()

        for i in D[0].columns:
            uniques = D[0][i].unique()
            for j in uniques:
                correspondingsplits[CART((D[0], D[1]), i, j)] = (i,j)
        # print("largest CART found: ", max(correspondingsplits), "with the attribute and value being", correspondingsplits.get(max(correspondingsplits)))
        return correspondingsplits.get(max(correspondingsplits))
    else:
        return (-1,-1)



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
    data = pd.read_csv(filename, header=None)
    # print("whole data: \n", data)
    X = data.iloc[: , :len(data.columns)-1]
    Y = data.iloc[: ,len(data.columns)-1:len(data.columns)]

    # print(CART(data,0,1))

    best_IG = bestSplit((X,Y), "IG")
    best_GINI = bestSplit((X,Y), "GINI")
    best_CART = bestSplit((X,Y), "CART")

    # print()
    return ((X,Y), best_IG, best_GINI, best_CART)

#testing
# load('train.txt')

# best_IG_fortrain = load('train.txt')
#
def classifyIG(train, test):
    """Builds a single-split decision tree using the Information Gain criterion
    and dataset train, and returns a list of predicted classes for dataset test
    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train
    Returns:
        A list of predicted classes for observations in test (in order)
    """
    data = pd.concat((test[0][0], test[0][1]), axis=1)

    # print(best_IG_fortrain[1])
    purityindex_D_y = wholeset_entropy(pd.DataFrame(splitter(pd.concat((train[0][0],train[0][1]), axis = 1), train[1][0], train[1][1])[0]).reset_index(drop = True), train[1][0])[3]
    purityindex_D_n = wholeset_entropy(pd.DataFrame(splitter(pd.concat((train[0][0],train[0][1]), axis = 1), train[1][0], train[1][1])[1]).reset_index(drop = True), train[1][0])[3]

    test_split = splitter(data, train[1][0], train[1][1])
    print()
    print("-------------------classification of IG------------------")
    print("predicted class: ", purityindex_D_y)
    print()
    print(pd.DataFrame(test_split[0]))
    print("classification error: ", error_index(pd.DataFrame(test_split[0]), purityindex_D_y))
    print()
    print("predicted class: ", purityindex_D_n)
    print()
    print(pd.DataFrame(test_split[1]))
    print("classification error: ", error_index(pd.DataFrame(test_split[1]), purityindex_D_n))


# classifyIG('train.txt', 'test.txt')

def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test
    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train
    Returns:
        A list of predicted classes for observations in test (in order)
    """
    data = pd.concat((test[0][0], test[0][1]), axis=1)

    purityindex_D_y = wholeset_entropy(pd.DataFrame(splitter(pd.concat((train[0][0], train[0][1]), axis=1), train[2][0], train[2][1])[0]).reset_index(drop = True), train[2][0])[3]
    purityindex_D_n = wholeset_entropy(pd.DataFrame(splitter(pd.concat((train[0][0], train[0][1]), axis=1), train[2][0], train[2][1])[1]).reset_index(drop = True), train[2][0])[3]

    test_split = splitter(data, train[2][0], train[2][1])
    print()
    print("-------------------classification of G------------------")
    print("predicted class: ", purityindex_D_y)
    print()
    print(pd.DataFrame(test_split[0]))
    print("classification error: ", error_index(pd.DataFrame(test_split[0]), purityindex_D_y))
    print()
    print("predicted class: ", purityindex_D_n)
    print()
    print(pd.DataFrame(test_split[1]))
    print("classification error: ", error_index(pd.DataFrame(test_split[1]), purityindex_D_n))

# classifyG('train.txt', 'test.txt')

def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test
    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train
    Returns:
        A list of predicted classes for observations in test (in order)
    """
    data = pd.concat((test[0][0], test[0][1]), axis=1)
    # print(best_IG_fortrain[1])

    purityindex_D_y = wholeset_entropy(pd.DataFrame(splitter(pd.concat((train[0][0], train[0][1]), axis=1), train[3][0], train[3][1])[0]).reset_index(drop=True), train[2][0])[3]
    purityindex_D_n = wholeset_entropy(pd.DataFrame(splitter(pd.concat((train[0][0], train[0][1]), axis=1), train[3][0], train[3][1])[1]).reset_index(drop=True), train[2][0])[3]

    test_split = splitter(data, train[3][0], train[3][1])
    print()
    print("-------------------classification of CART------------------")
    print("predicted class: ", purityindex_D_y)
    print()
    print(pd.DataFrame(test_split[0]))
    print("classification error: ", error_index(pd.DataFrame(test_split[0]), purityindex_D_y))
    print()
    print("predicted class: ", purityindex_D_n)
    print()
    print(pd.DataFrame(test_split[1]))
    print("classification error: ", error_index(pd.DataFrame(test_split[1]), purityindex_D_n))

# classifyCART('train.txt', 'test.txt')

def main():
    """This portion of the program will run when run only when main() is called.
    This is good practice in python, which doesn't have a general entry point
    unlike C, Java, etc.
    This way, when you <import HW2>, no code is run - only the functions you
    explicitly call.
    """
    # for testing or any other purposes please change the path of file here for both train and test data.
    train_data = load('train.txt')
    test_data = load('test.txt')

    info_prompter(train_data) # prints the informaton prompt.

    # finds the classification of IG, G, CART
    classifyIG(train_data, test_data)
    classifyG(train_data , test_data)
    classifyCART(train_data , test_data)
    print()

if __name__ == "__main__":
    """__name__=="__main__" when the python script is run directly, not when it
    is imported. When this program is run from the command line (or an IDE), the
    following will happen; if you <import HW2>, nothing happens unless you call
    a function.
    """
    print("-----------------working on best splits and classiying data------------------")
    main()
'''------------------------task 1 -----------------'''
from functools import reduce
import re

def test_print():
    print("This is a test statement.")

'''---------------- task 2 -------------------'''
def list_set_length():
    items_list = [1, 2, 3, 4, 5, 3, 2, 1, 5]
    items_set = {1, 2, 3, 4, 5, 3, 2, 1, 5}
    print()
    print("------------task 2------------------")
    print("the length of list is : ", len(items_list))
    print("the length of set is : ", len(items_set))

'''---------------- task 3 -------------------'''
def set_difference():
    s = {1,2,3,4,5,6}
    t = {4,5,6,7,8,9}
    output = {x for x in s if x not in t}
    print()
    print("------------task 3------------------")
    print("the resulting set difference: [s-t]", output)

'''---------------- task 4 -------------------'''
def three_tuples():
    integers_set = {-4,-2,1,2,3,5,0}
    divisibleby3 = {(i,j,k) for i in integers_set for j in integers_set for k in integers_set
                    if (i+j+k)%3==0
                    }
    print()
    print("------------task 4------------------")
    print("set of divisible by 3: \n", divisibleby3 )
    print("length of the new set divisible by 3: ", len(divisibleby3))

'''---------------- task 5 -------------------'''
def dict_init():
    mydict = {'Hopper': 'Grace', 'Einstein': 'Albert', 'Turing': 'Alan', 'Lovelace': 'Ada'}
    print()
    print("------------task 5------------------")
    print("initialized dictonary:", mydict)

'''declaration of a dlist'''
mydict = {'Hopper': 'Grace', 'Einstein': 'Albert', 'Turing': 'Alan', 'Lovelace': 'Ada'}
mydict2 = {'sai': 'vishnu', 'anudeep': 'kadiyala'}
dlist = [mydict, mydict2]
key = 'Hopper';
key2 = 'sai'
key3 = 'sa'
def dict_find(dlist, key):
    output = next((item for item in dlist if key in list(item.keys())), 'NOT PRESENT')
    print()
    print("given key, ", key, "is searched in the dictonary list and found : ",output)

givenword1 = 'with'
givenword2 = 'the'
givenword3 = 'sai'

'''---------------- task 6 -------------------'''
def file_word_count(givenword):
    number_of_occ = 0;
    with open(r'stories.txt', 'r') as file:
        given = file.read()
        eachline = given.upper().split()

        for word in eachline:
            if word == givenword.upper():
                number_of_occ += 1
    print()
    print("number of occurences of given: ", givenword, " is/are: ", number_of_occ);

'''--------------------task 7---------------------'''
#reads the given file and creates a dictionary that have documents with key index values as 0, 1, 2...
def read_file_make_strlist(filename):
    with open(filename) as file:
        given = file.readlines()
        key= 0;
        dict_list = {}
        for line in given:
            line = line.rstrip("\n")
            dict_list[key] = line
            key += 1
        # print(dict_list)
    return dict_list

strlist = read_file_make_strlist('stories.txt') # creating a dictionary list for stories.txt

# to split each line of document and tokenize the line
def tokenizer(strlist):
    dict_tokened = {}
    for i in range(0,len(strlist)):
        dict_tokened[i] = strlist[i].split()
    return dict_tokened

# print(tokenizer(strlist))

# creates the datastructure using the tokenized strlist and counts their occurence lines.
def make_inverse_index(strlist):
    print()
    print("-------------------task 7---------------------------")
    dict_all_occurences = {}
    givedict = tokenizer(strlist) #tokenizer called
    for i in givedict: # picks a line
        # print(givedict[i])
        for word in givedict[i]: # chooses a word of a particular line
            if word.upper() in dict_all_occurences: # if there in the dictionary of occurences,
                if(i in dict_all_occurences.get(word.upper())): #compares the key and value and check if exits if exists then it will not be appened.
                    continue;
                else: dict_all_occurences[word.upper()].append(i); #appends it if in the occurence dictionary with new line number
            else:
                dict_all_occurences[word.upper()] = [i] # if a key value doesnt exist then it will be added to the occurence

    # print(dict_all_occurences)
    return dict_all_occurences




dict_occurences = make_inverse_index(strlist)
# for a particular strlist the occurence list is called.

# or search
def or_search(dict_occurences, word):
    wordlist = word.upper().split()
    result = []
    for a in wordlist:
        if a in dict_occurences.keys():
            result.append(dict_occurences.get(a))
        else:
            continue
        # print(a, result)

    result = list(set(x for l in result for x in l)) # it will create a set and returns unique values from the result of occurence of given word.
    print("for the given word,", word,"or search result: ", result)
    return result

# and search
def and_search(dict_occurences, word):
    wordlist = word.upper().split()
    result = []
    for a in wordlist:
        if a in dict_occurences.keys():
            result.append(dict_occurences.get(a))
        else:
            continue
    result = list(reduce(lambda i,j: i&j, (set(x) for x in result))) #gives out unique and common values if exists
    print("for the given word,", word,"and search result: ", result)
    return result


# write the output of the occurences to the result.txt
def to_write_to_file():
    # results of asked words
    or_NY_result = or_search(dict_occurences, 'New York')
    and_NY_result=  and_search(dict_occurences, 'New York')
    or_HC_result = or_search(dict_occurences, 'Health Care')
    and_HC_result = and_search(dict_occurences, 'Health Care')
    # print(or_NY_result)

    # opens a result.txt file and writes to it
    with open('result.txt', 'w') as file:
        file.write('or result of NY:' + '\n')
        file.write('\n')
        # loops through the result and writes to the file
        for temp in or_NY_result:
            file.write('%s'%temp+', ');
        file.write('\n')
        file.write('\n')

        file.write('and result of NY:' + '\n')
        # loops through the result and writes to the file
        for temp in and_NY_result:
            file.write('%s'%temp+', ');
        file.write('\n')
        file.write('\n')

        file.write('or result of Health care:' + '\n')
        # loops through the result and writes to the file
        for temp in or_HC_result:
            file.write('%s'%temp+', ');
        file.write('\n')
        file.write('\n')

        file.write('and result of Health care:' + '\n')
        # loops through the result and writes to the file
        for temp in and_HC_result:
            file.write('%s'%temp+', ');
        file.write('\n')
        file.write('\n')
        file.write('--End of file--')

test_print()
list_set_length()
set_difference()
three_tuples()
dict_init()
dict_find(dlist, key)
dict_find(dlist, key2)
dict_find(dlist, key3)
print()
print("------------task 6------------------")
file_word_count(givenword1)
file_word_count(givenword2)
file_word_count(givenword3)
make_inverse_index(strlist)
to_write_to_file()
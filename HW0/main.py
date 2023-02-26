#1
def test_print():
    print("This is test")
if _name=='main_':
    test_print()

#2
print("Question 2")
def list_set_length():
    items_list=[1,2,3,4,5,3,2,1,5]
    items_set={1,2,3,4,5,3,2,1,5}
    print("length of items list: ",len(items_list))
    print("length of items set:",len(items_set))
list_set_length();

#3
print("Question 3")
def set_difference():
    S = {1, 2, 3, 4, 5, 6}
    T = {4, 5, 6, 7, 8, 9}
    return [i
            for i in S
            if i not in T]
print(set_difference());

#4
print("Question 4")
def three_tuples():
    S = {-4, -2, 1, 2, 3, 5, 0}
    return [
        (i, j, k)
        for i in S
        for j in S
        for k in S
        if (i+j+k) % 3 == 0]
print(three_tuples());

#5a
print("Question 5a")
def dict_init():
    mydict = {'Hopper': 'Grace', 'Einstein': 'Albert', 'Turing': 'Alan', 'Lovelace': 'Ada'}
    print(mydict)
dict_init()

#5b
print("Question 5b")
dlist = [{'Hopper': 'Grace', 'Einstein': 'Albert', 'Turing': 'Alan', 'Lovelace': 'Ada'}, {'ABC': 'def', 'GHI': 'jkl', 'MNO': 'pqr', 'STU': 'vwx'}, {'qw': 'er', 'ty': 'ui'}]
k = input('5th b - entry for search\n')
def dict_find(dlist,k):
        lis=[d for d in dlist if k in d]

        if not lis:
            print("NOT FOUND")
        else:
            print(lis)
dict_find(dlist,k)

#6
print("Question 6")
def file_word_count():
    f1=open('stories.txt')
    l1=f1.readlines()
    ct=0
    req = 'new'
    req1=req.lower()
    for x in l1:
        y = x.lower().split()
        for z in y:
            if z==req1:
                ct=ct+1
    print("for word ",req," the count is:",ct)
file_word_count();

# print("Question 7a")
strlist=open('stories.txt', 'r')
inverseIndex={}
def make_inverse_index(f):
    l1 = f.readlines()
    ct = 0
    for x in l1:
        ct = ct + 1
        y = x.lower().split()
        for q in y:
            if q not in inverseIndex:
                inverseIndex[q] = [ct - 1]
            else:
                inverseIndex[q].append(ct - 1)
    return [inverseIndex]
make_inverse_index(strlist);
#here the inverse index is not printed because the inverse index size is too large.
#incase you want to check the dictionary you can uncomment the below line and run it.
#print(make_inverse_index(strlist));
req=input("Entry for search in OR and AND SEARCH")
print("Question 7b")
def or_search(inverseIndex, query):
    xyz=[]
    y=query.split()
    for q in y:
        if q in inverseIndex:
            z=(inverseIndex[q])
            for i in z:
                if i not in xyz:
                    xyz.append(i)
    return xyz
print(or_search(inverseIndex,req));
print("Question 7c")
def and_search(inverseIndex, query):
    res=[]
    y=query.split()
    for q in y:
        xyz = []
        if q in inverseIndex:
            z = (inverseIndex[q])
            for i in z:
                if i not in xyz:
                    xyz.append(i)
        res.append(xyz)
    output=list(set.intersection(*map(set, res)))
    return output
print(and_search(inverseIndex,req))

print("question 7d")
print("searching new york")
q1=or_search(inverseIndex,'new york')
q2=and_search(inverseIndex,'new york')
print(q1)
print(q2)
print("searching health care")
q3=or_search(inverseIndex,'health care')
q4=and_search(inverseIndex,'health care')
print(q3)
print(q4)
f=open("results.txt",'w')
f.write("searching new york\n")
f.write("OR Search:\n[")
for p1 in q1:
    f.write("%s" %p1)
    f.write(' ')
f.write("]\nAND Search:\n[")
for p1 in q2:
    f.write("%s" %p1)
    f.write(' ')
f.write("]\nsearching health care\n")
f.write("OR Search:\n[")
for p1 in q3:
    f.write("%s" %p1)
    f.write(' ')
f.write("]\nAND Search:\n[")
for p1 in q4:
    f.write("%s" %p1)
    f.write(' ')
f.write("]")
f.close()
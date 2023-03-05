import math

print('-------------------general mean and variance ---------------------')
dict_data = {0: 60, 1:720, 2:1200, 3:750, 4:240, 10:30}
mean = 0
variance = 0
sum = 0
dict_numofcars = []

for i in list(dict_data.keys()):
    sum += i*dict_data.get(i) # total number of cars
    dict_numofcars.append(i*dict_data.get(i))

print("sum of all the cars ", sum)
mean = sum/3000 # total no of cars / number of families
print("total number of cars: ", dict_numofcars)
print("number of cars per family: ", mean)

for i in list(dict_data.keys()):
    variance += math.pow(((i-2.21)),2)*dict_data.get(i)
# print(variance)
print("variance of the data: ", variance/3000)


print()
print("-------------mean and variance without 10-----------------------")


dict_data1 = {0: 60, 1:720, 2:1200, 3:750, 4:240}
mean2 = 0
variance2 = 0
sum = 0
dict_numofcars = []

for i in list(dict_data1.keys()):
    sum += i*dict_data1.get(i) # total number of cars
    dict_numofcars.append(i*dict_data1.get(i))

print("sum of all the cars ", sum)
mean2 = sum/(3000-30) # total no of cars / number of families
print("total number of cars: ", dict_numofcars)
print("number of cars per family: ", mean2)

for i in list(dict_data1.keys()):
    variance2 += math.pow(((i-2.21)),2)*dict_data1.get(i)
# print(variance2)
print("variance of the data without 10: ", variance2/(3000-30))

print()

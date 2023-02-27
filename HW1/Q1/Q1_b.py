import math

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
print(dict_numofcars)
print("number of cars per family: ", mean)

for i in list(dict_data.keys()):
    variance += math.pow(((i-2.21)),2)*dict_data.get(i)
print(variance)
print(variance/3000)

prob = []
for i in list(dict_data.keys()):
    prob.append(dict_data.get(i)/3000)
print(prob)

vari = 0
for i in list(dict_data.keys()):
    temp = 0
    if(i == 10): temp = prob[5]
    else: temp = prob[i]
    vari += math.pow(((i-2.21)),2)*temp

print(vari)


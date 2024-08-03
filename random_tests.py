import numpy as np
import random

myList = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
minValue = min(myList)

myList = [x - minValue for x in myList]
#print(myList)
#print()

myArray = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
myArray = np.array([])
#print(myArray)
#print()

random_values = [round(random.uniform(0,1e9),2) for _ in range(14)]
#print(random_values)

myList = [2, 12, 10, 30]
my2ndList = [9, 7, 3, 33]

myList[-3] = my2ndList[-3]

del myList[3]
myList.insert(3, 49)
print(myList)
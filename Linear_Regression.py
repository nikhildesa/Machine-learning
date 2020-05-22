# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:02:49 2020

@author: nikhil
"""

# Run The code from Training line 29


import numpy as np

filename = 'BikeAll.txt'
data = np.loadtxt(filename,skiprows = 1)
np.take(data,np.random.permutation(data.shape[0]),axis=0,out = data)
train = data[0:439,:]
test = data[439:585,:]
val = data[585:,:]

np.savetxt('Desai_Nikhil_Train.txt', train, fmt="%0.2f",delimiter='\t',header = "439\t11",comments = "")
np.savetxt('Desai_Nikhil_Test.txt', test, fmt="%0.2f",delimiter='\t',header = "146\t11",comments = "")
np.savetxt('Desai_Nikhil_Valid.txt', val, fmt="%0.2f",delimiter='\t',header = "146\t11",comments = "")






# Training
import numpy as np
filename = input("Enter training set file")
x = np.loadtxt(filename,skiprows = 1)
y = x[:,11]    
x = x[:,:-1]
M = np.ones([len(x),1])
a = np.concatenate((M,x),axis = 1)

# model 1
w1 = np.dot(np.linalg.pinv(np.dot(a.T, a)),np.dot(a.T, y))
intttm = np.dot(a,w1) - y
intm = intttm * intttm
J1 = np.dot(M.T,intm)/(2*len(a))
print("\nweights for Linear model",w1)
print("\nJ value Linear model is ",J1)

# model 2
x_square = np.square(x)
b = np.concatenate((M,x),axis = 1)
b = np.concatenate((b,x_square),axis = 1)
w2 = np.dot(np.linalg.pinv(np.dot(b.T, b)),np.dot(b.T, y))
intttm = np.dot(b,w2) - y
intm = intttm * intttm
J2 = np.dot(M.T,intm)/(2*len(b))
print("\nweights for Quadratic model",w2)
print("\nJ value Quadratic model",J2)

# model 3
x_cube = np.power(x,3)
c = np.concatenate((M,x),axis = 1)
c = np.concatenate((c,x_cube),axis = 1)
w3 = np.dot(np.linalg.pinv(np.dot(c.T, c)),np.dot(c.T, y))
intttm = np.dot(c,w3) - y
intm = intttm * intttm
J3 = np.dot(M.T,intm)/(2*len(c))
print("\nweights for cubic model",w3)
print("\nJ value Cubic model",J3)




#<--------------------------------------->
#Validation

# Validation
import numpy as np
filename = input("Enter validation file\n")	
x = np.loadtxt(filename,skiprows = 1)
y = x[:,11]    
x = x[:,:-1]
M = np.ones([len(x),1])


# model 1
a = np.concatenate((M,x),axis = 1)
#w1 = np.dot(np.linalg.pinv(np.dot(a.T, a)),np.dot(a.T, y))
intttm = np.dot(a,w1) - y
intm = intttm * intttm
J1 = np.dot(M.T,intm)/(2*len(a))
print("J value Linear model is \n",J1)

# model 2
x_square = np.square(x)
b = np.concatenate((M,x),axis = 1)
b = np.concatenate((b,x_square),axis = 1)
#w2 = np.dot(np.linalg.pinv(np.dot(b.T, b)),np.dot(b.T, y))
intttm = np.dot(b,w2) - y
intm = intttm * intttm
J2 = np.dot(M.T,intm)/(2*len(b))
print("J value Quadratic model \n",J2)

# model 3
x_cube = np.power(x,3)
c = np.concatenate((M,x),axis = 1)
c = np.concatenate((c,x_cube),axis = 1)
#w3 = np.dot(np.linalg.pinv(np.dot(c.T, c)),np.dot(c.T, y))
intttm = np.dot(c,w3) - y
intm = intttm * intttm
J3 = np.dot(M.T,intm)/(2*len(c))
print("J value Cubic model \n",J3)




# Testing
#<------------------------------------------------>
import numpy as np
filename = input("Enter test file")
x = np.loadtxt(filename,skiprows = 1)
y = x[:,11]    
x = x[:,:-1]
M = np.ones([len(x),1])
a = np.concatenate((M,x),axis = 1)

# model 1
#w1 = np.dot(np.linalg.pinv(np.dot(a.T, a)),np.dot(a.T, y))
intttm = np.dot(a,w1) - y
intm = intttm * intttm
J1 = np.dot(M.T,intm)/(2*len(a))
print("\nJ value Linear model is ",J1)


y_mean = np.mean(y)
denom =  float((np.dot(M.T,((y-y_mean) ** 2)))/ (2*len(a)))

r_square1 = 1 - (J1 / denom)
Adjusted_r_square1 = 1 - (((1 - r_square1) * (len(a) - 1))/(len(a) - a.shape[1] - 1))
print("\nAdjusted r2 for Linear model is ",Adjusted_r_square1)





# model 2
x_square = np.square(x)
b = np.concatenate((M,x),axis = 1)
b = np.concatenate((b,x_square),axis = 1)
#w2 = np.dot(np.linalg.pinv(np.dot(b.T, b)),np.dot(b.T, y))
intttm = np.dot(b,w2) - y
intm = intttm * intttm
J2 = np.dot(M.T,intm)/(2*len(b))
print("\nJ value Quadratic model",J2)


r_square2 = 1 - (J2 / denom)
Adjusted_r_square2 = 1 - (((1 - r_square2) * (len(a) - 1))/(len(a) - b.shape[1] - 1))
print("\nAdjusted r2 for Quadratic model is ",Adjusted_r_square2)




# model 3
x_cube = np.power(x,3)
c = np.concatenate((M,x),axis = 1)
c = np.concatenate((c,x_cube),axis = 1)
#w3 = np.dot(np.linalg.pinv(np.dot(c.T, c)),np.dot(c.T, y))
intttm = np.dot(c,w3) - y
intm = intttm * intttm
J3 = np.dot(M.T,intm)/(2*len(c))
print("\nJ value Cubic model",J3)

r_square3 = 1 - (J3 / denom)
Adjusted_r_square3 = 1 - (((1 - r_square3) * (len(a) - 1))/(len(a) - c.shape[1] - 1))
print("\nAdjusted r2 for Cubic model is ",Adjusted_r_square3)




















# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:56:48 2020

@author: nikhi
"""

l = []
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:38:29 2020

@author: nikhi
"""
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

def update_centroid(flag,data):
    a = b = c = d = 0
    for i in range(0,len(flag)):
        if(flag[i] == 1):
            a+=(data[i,0]/flag.count(1))
            b+=(data[i,1]/flag.count(1))
        else:
            c+=(data[i,0]/flag.count(2))
            d+=(data[i,1]/flag.count(2))
    return a,b,c,d

def eculidean(p,c):
    return ((p[0] - c[0])**2 + (p[1] - c[1])**2) 

def kmeans(data,centroid):
    flag = []
    a = centroid[0,0]
    b = centroid[0,1]
    c = centroid[1,0]
    d = centroid[1,1]
    for loop in range(0,100):
        flag.clear()
        for i in range(0,len(data),1):
            x = data[i,0]
            y = data[i,1]
            dist1 = sqrt((x - a)**2 + (y - b)**2)
            dist2 = sqrt((x - c)**2 + (y - d)**2)
    
            if(dist1 < dist2):
                flag.append(1)
            else:
                flag.append(2)
    
        a,b,c,d = update_centroid(flag,data)
    j = 0
    for i in range(0,len(flag)):
        if(flag[i] == 1):
            points = (data[i,0],data[i,1])
            centre = (a,b)
            j+=eculidean(points,centre)
        else:
            points = (data[i,0],data[i,1])
            centre = (c,d)
            j+=eculidean(points,centre)
    j=j/len(data)      
    return a,b,c,d,flag,j


#file_train = "P4Data.txt"
file_train = input("Enter a file with points")
data = np.loadtxt(file_train,skiprows = 1)

#file_centroid = "P4Centroids.txt"
file_centroid = input("Enter a file with centroid")
centroid = np.loadtxt(file_centroid,skiprows = 1)

print("c1 is ",centroid[0][0],",",centroid[0][1])
print("c2 is ",centroid[1][0],",",centroid[1][1])

a,b,c,d,flag,j = kmeans(data,centroid)

figure(num=None, figsize=(8, 4), dpi=80)
plt.title("data") 
plt.xlabel("x axis") 
plt.ylabel("y axis") 
plt.plot(data[:,0],data[:,1],"ob") 
plt.plot(centroid[0,0],centroid[0,1], marker="x",markersize=8)
plt.plot(centroid[1,0],centroid[1,1], marker="x",markersize=8)
plt.show() 

figure(num=None, figsize=(8, 4), dpi=80)
plt.scatter(data[:,0],data[:,1],c = flag,)
plt.plot(a,b, marker="^",c = "purple",markersize=15)
plt.plot(c,d, marker="^",c = "goldenrod",markersize=15)
plt.show()

print("c1 is ",a,",",b)
print("c2 is ",c,",",d)

print("j is ",j)
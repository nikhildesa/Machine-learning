
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 13:58:47 2020

@author: Nikhil Desai


"""
from matplotlib import pyplot as plt
def init_weight(col_x):
	w = np.zeros([col_x,1])
	return w

def sigmoid(X,w):
	return 1/(1 + np.exp(-np.dot(X,w)))


def model_predict(X,y,w,m,alpha,iteration,ones):
	l = []
	for i in range(iteration):
		H = sigmoid(X,w)
		cost = -y * np.log(H) - (1-y) * np.log(1-H)
		j = (1/len(m)) * np.dot(ones,cost)
		l.append(j)
		temp1 = alpha * np.dot(np.subtract(H,y).T,X)
		w = w - temp1.T
		
		
		
	
	return w,H,l,j,cost
	


import numpy as np
#filename = "DivorceAll.txt"
#data = np.loadtxt(filename,skiprows = 1)
#np.take(data,np.random.permutation(data.shape[0]),axis=0,out = data)
#ans = int((data.shape[0] * 80)/100)
#train = data[0:ans,:]
#test = data[ans:,:]

#np.savetxt('Desai_Nikhil_Train.txt', train, fmt="%0.2f",delimiter='\t',header = "136\t54",comments = "")
#np.savetxt('Desai_Nikhil_Test.txt', test, fmt="%0.2f",delimiter='\t',header = "34\t54",comments = "")

file = input("Enter Train file\t")
X = np.loadtxt(file,skiprows = 1)
y = X[:,-1]
y = np.reshape(y, (136, 1))
X = X[:,:-1]
m = np.ones([len(X),1])
X = np.concatenate((m,X),axis = 1)
w = init_weight(X.shape[1])
alpha = 0.0001
iteration = 5000
ones = np.ones([1,X.shape[0]])

w,H,l,j,cost = model_predict(X,y,w,m,alpha,iteration,ones)

l = np.reshape(l,(5000,1))
iterations = list(range(1,len(l)+1))
plt.title("iterations vs j(w) for training")
plt.xlabel("Iterations")
plt.ylabel("j(w)")
plt.plot(iterations,l)

#< Testing >



file_test = input("Enter Test file\t")
X_test = np.loadtxt(file_test,skiprows = 1)
y_test = X_test[:,-1]
y_test = np.reshape(y_test, (34, 1))
X_test = X_test[:,:-1]
m_test = np.ones([len(X_test),1])
X_test = np.concatenate([m_test,X_test],axis = 1)
H_test = sigmoid(X_test,w)

ones_test = np.ones([1,X_test.shape[0]])


cost_test = -y_test * np.log(H_test) - (1-y_test) * np.log(1-H_test)
j_test = (1/len(m_test)) * np.dot(ones_test,cost_test)
y_pred = np.round(H_test)

print("Final j is",j_test)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

print("tp is ",tp)
print("tn is ",tn)
print("fp is ",fp)
print("fn is ",fn)

cm = confusion_matrix(y_test,y_pred)
accuracy = (tp+tn)/(tp+tn+fp+fn)
print("accuracy is ",accuracy)

precision = tp/(tp+fp)
print("precision is ",precision)

recall = tp/(tp+fn)
print("recall is",recall)

f1 = (2 * precision * recall)/(precision + recall)
print("f1 score is ",f1)







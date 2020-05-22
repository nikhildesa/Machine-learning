import csv
import pandas as pd
import matplotlib.pyplot as plt

with open ('_IrisData.txt', 'r') as f:
    first_column = [row[0] for row in csv.reader(f,delimiter='\t')]
    #print (first_column)
with open ('_IrisData.txt', 'r') as f:
    third_column = [row[2] for row in csv.reader(f,delimiter='\t')]
with open ('_IrisData.txt', 'r') as f:
    last_column = [row[4] for row in csv.reader(f,delimiter='\t')]
mapping = {'versicolor' : 'o', 'virginica': 'x', 'setosa': '+'}

df1 = pd.DataFrame(first_column) 
df2 = pd.DataFrame(third_column) 
df3 = pd.DataFrame(last_column) 

df1.columns = ['sepallength']
df2.columns = ['petallength']
df3.columns = ['species']

dataset = pd.concat([df1, df2,df3], axis=1, sort=False)
dataset["sepallength"] = pd.to_numeric(dataset["sepallength"])
dataset["petallength"] = pd.to_numeric(dataset["petallength"])

import seaborn as sns
sns.lmplot(x = "sepallength",y = "petallength",data = dataset,hue = "species",fit_reg=False,legend=False)
plt.legend()
plt.savefig('Desai_Nikhil_MyPlot')
plt.title("Length comparison")
plt.show()


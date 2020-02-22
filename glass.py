# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:17:17 2019

@author: Ganesh
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
glass = pd.read_csv("F:\\Python\\Machine learning\\KNearestNeighbour\\glass.csv")

#finding na's in dataset

glass.isna().sum()
#there is no na's in dataset

#performing feature Engineering

glass.describe()
glass.head()
glass.tail()
glass.info()
glass.shape

#every variable is a numeric type of data

glass.columns

#find skewness and kurtosis
glass.skew()
glass.kurt()

#visualizing variables
sns.distplot(glass.Type)

#more number of glass in second type and first type

#checking distplot for glass.K because of highest positive kurtosis

sns.distplot(glass.K)
plt.hist(glass.K)
plt.boxplot(glass.K)
plt.boxplot(glass.Ba)
plt.boxplot(glass.RI)
#there are so many outliers in Every variable variable

#normalizing the datasets
from sklearn import preprocessing

numerical_Data = glass.iloc[:,0:9]

norm_data = preprocessing.normalize(numerical_Data)

z = pd.DataFrame(norm_data)

#splitting the data into train and test
from sklearn.model_selection import train_test_split

X = z
Y = glass.iloc[:,9]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)
#building a model

from sklearn.neighbors import KNeighborsClassifier as KNC
acc = []
for i in range(3,50,2):
    neigh = KNC(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train)==Y_train)
    test_acc = np.mean(neigh.predict(X_test)==Y_test)
    acc.append([train_acc,test_acc])

acc = pd.DataFrame(acc)
plt.plot(np.arange(3,50,2),acc.iloc[:,0], "bo-")
plt.plot(np.arange(3,50,2), acc.iloc[:,1], "ro-")
plt.legend(["acc.iloc[:,0]"], ["acc.iloc[:,1]"])

#best model while giving K = 5


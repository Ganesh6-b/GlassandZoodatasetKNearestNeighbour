# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:25:13 2019

@author: Ganesh
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as KNC

dataset = pd.read_csv("F:\\Python\\Machine learning\\KNearestNeighbour\\Zoo.csv")

dataset["animal name"].value_counts()

df = dataset.iloc[:,1:]

#feature engineering

#finding na's

dataset.isna().sum()

#there is no na's in this dataset

df.head()

df.describe()

df.info()

#correlations

correlation = df.iloc[:,0:17].corr()

sns.heatmap(correlation)

#splitting the data set

X = df.iloc[:,0:16]
Y = df.iloc[:,16]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

acc = []

for i in range(3,50,2):
    neigh = KNC(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train)==Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

acc = pd.DataFrame(acc)
acc = acc.rename(columns = {0 : "Training_accuracy", 1 : "Testing_accuracy"})

plt.plot(np.arange(3,50,2), acc.Training_accuracy, "bo-")
plt.plot(np.arange(3,50,2), acc.Testing_accuracy, "ro-")
plt.legend(["acc.Training_accuracy"],["Testing_accuracy"])

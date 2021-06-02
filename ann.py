#!/usr/bin/env python
# coding: utf-8

# # Data Processing

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  tensorflow as tf

# Import dataset
dataset = pd.read_csv("Churn_Modelling.csv")

dataset.head()

# Dependent Variable
x = dataset.iloc[:, 3: -1].values

#Independent Variable
y = dataset.iloc[:, -1].values

print (pd.DataFrame(x))


print(y)


# ### Encoding Categorical data

# ##### Encoding Independent Columns



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# Gender column




x[:,2] = le.fit_transform(x[:,2])

#Print x as a dataframe
print(pd.DataFrame(x))


# One hot Encoding of Geography Column


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder="passthrough")

x = np.array(ct.fit_transform(x))

print(pd.DataFrame(x))


# #### Spliiting Data Into Train and Testing Sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size  = 0.2, random_state = 0)

# #### Feature Scaling
# 


#feature sclaing
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


print(pd.DataFrame(x_train))


# ## Building The ANN

# #### Initializing the ANN



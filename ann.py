# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  tensorflow as tf

dataset = pd.read_csv("Churn_Modelling.csv")

x = dataset.iloc[:, 3: -1].values
y = dataset.iloc[:, -1].values

print (x)
print (y)
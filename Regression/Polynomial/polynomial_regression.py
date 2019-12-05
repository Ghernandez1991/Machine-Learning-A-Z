#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set



data = pd.read_csv('Position_Salaries.csv')
#we need to get all the independent variable data- use iloc to grab what is needed 

X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values





# since we have such a small data set(10 values), we are not going to test and train. We need the model 
#to use all of the available data points to make a predicition. Spliting them would reduce them down. 
"""from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""


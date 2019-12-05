#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set



data = pd.read_csv('Data.csv')
#we need to get all the independent variable data- use iloc to grab all the rows, and all but the last column

X = data.iloc[:,:-1].values
Y = data.iloc[:, 3].values



#this library allows us to deal with the missing numbers. Lots of classes and methods useful for machine learning preprocessing

from sklearn.preprocessing import Imputer
#create an object of the class, our missing values are NAN, we want to take the average of the column
# and we want the colum mean so axis is 0
Imputer= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# we fit the model to x data sext, we want all the rows, and the second and 3rd column
Imputer  = Imputer.fit(X[:, 1:3])
# we want to take those mean values and apply them to the Nan that are in the respective columsn
X[:, 1:3] = Imputer.transform(X[:, 1:3])    

# encoding the catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#create variable and call the label encoder class-- we then select the column needed 
LabelEncoder_X = LabelEncoder()
#we apply fit transform the all the rows and the column we need- in this case country- it will encode the catagories numerical- we put X[:, 0] at the begining because we want to put the encoded values back into the array
X[:, 0]= LabelEncoder_X.fit_transform(X[:, 0])
X
#note- the equation would make you beleive that spain and germany have a higher value than france(0). This is not true. We need to do dummy encoding to make them equal
#call categorical_features on the categorical column(country)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X
#we apply the same logic to the Y variable(purchased) Since it is only 1 column, we do not need to specify like we did in X
LabelEncoder_Y = LabelEncoder()
Y= LabelEncoder_Y.fit_transform(Y)


# we need to split data into training and testing set. The model will learn from the training set and then test it on the testing set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#we need to scale our features to ensure they are evenly represented in the model. As salary is many times larger than age, we dont want that to dominate our model. 
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
# we want to fit the object to the training set, and then transform it
X_train = sc_X.fit_transform(X_train)
# we dont need to fit on the test set because it is already fit in the training set
X_test = sc_X.transform(X_test)
#you dont necessarily need to scale your dummy variables, although in certain circumstances you may need/want to
# we dont need to scale our y value because it is categorical and a value betwen 0-1. In some regression models and others, you will need to scale your Y as well. 


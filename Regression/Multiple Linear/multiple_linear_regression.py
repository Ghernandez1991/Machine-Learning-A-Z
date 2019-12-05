import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set



data = pd.read_csv('50_Startups.csv')
#we need to get all the independent variable data- use iloc to grab all the rows, and all but the last column

X = data.iloc[:,:-1].values
Y = data.iloc[:, 4].values

#we must encode the categorical state variable. 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LabelEncoder_x = LabelEncoder()
#grab the categorical variable column, in this case the third index
X[:, 3] = LabelEncoder_x.fit_transform(X[:, 3])
#categorical variables are listed in column three
onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()



#avoiding the dummy variable trap, you dont take the first column 0, so you take everything from index1 to the end

X = X[:, 1:]



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#we do not need to do feature scaling for MLR as the library takes care of it for us. 
from sklearn.linear_model import LinearRegression
#take the linearRegression class, and create an object of it. regressor
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

#use backward elimination to determine which independent variables are most significant and most useful to our model. 
import statsmodels.api as sm
#take matrix of x and add a column of 1s. We need to associate this to our constant of b0 in the formula
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#we want to create a matrix of optimal features that most influence the dependent variable 
#grab all the columns
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

#select our significance level. 
#SL = 0.05
#call the sm class abd select ordinary least squares. Then fit the model
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

#find the predictor with the higest p value
regressor_OLS.summary()
#x3 has the highest P value. we are going to remove it from the X array


#note we removed third column
X_opt = X[:, [0, 1, 2, 4, 5]]

#select our significance level. 
#SL = 0.05
#call the sm class abd select ordinary least squares. Then fit the model
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

#find the predictor with the higest p value
regressor_OLS.summary() 

#x3 has the highest p value, so remove it. 
X_opt = X[:, [0, 1,  4, 5]]

#select our significance level. 
#SL = 0.05
#call the sm class abd select ordinary least squares. Then fit the model
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

#find the predictor with the higest p value
regressor_OLS.summary()


#x3 has the highest p value, so remove it. 
X_opt = X[:, [0, 1,  4]]

#select our significance level. 
#SL = 0.05
#call the sm class abd select ordinary least squares. Then fit the model
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

#find the predictor with the higest p value
regressor_OLS.summary()

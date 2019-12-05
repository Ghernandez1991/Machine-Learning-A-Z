#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the data set



data = pd.read_csv('Salary_Data.csv')
#we need to get all the independent variable data- use iloc to grab all the rows, and all but the last column

X = data.iloc[:,:-1].values
Y = data.iloc[:, 1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


# fit the simple linear regression to the training set

from sklearn.linear_model import LinearRegression
#we call the linearregression from the class
regressor = LinearRegression() 
#fit the xtrain and y train to the object regressor. it takes x, y as parameters
regressor.fit(X_train, Y_train)


#prediciting the test set results
#create a vector of predicitions 
y_pred = regressor.predict(X_test)
#you can then look at y_pred and compare to y_test to see how closely the model is to 
#our actual y values

#visualize the training set results-- note that this is based on the training set
# our model did not 'learn' on the test set. only on the training set

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience(training set)")
plt.ylabel("Salary")
plt.xlabel("Years of experience ")
plt.show()


#lets visualize the testing set results. 
# note that the model was built with the training information
#the test set was not used to 'build' the model. so we can look at the model'blue line'
#and see how those values compare to the new test values 'red dots' it has not see before


plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience(test set)")
plt.ylabel("Salary")
plt.xlabel("Years of experience ")
plt.show()

#we can make new predictions for the Y variable(salary)
#you must make the array the same size as the one used to train the model 
new_employee_experience = np.array([[1]])
new_employee_salary = regressor.predict([[1]])





# In this program, we use a logistic regression model to predict which passengers survived the titanic calamity.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the data.

passengers = pd.read_csv('passengers.csv')
# print(passengers.head())

# As follows, we will determine some columns to be the features of our model.

# Updating sex column to numerical.

passengers['Sex'] = passengers.apply(lambda row: 1 if (row['Sex'] == 'female') else 0, axis = 1)

# Filling the nan values in the age column.

passengers['Age'] = passengers['Age'].fillna(value = np.mean(passengers['Age']))

# Creating a first class column.

passengers['FirstClass'] = passengers.apply(lambda row: 1 if (row['Pclass'] == 1) else 0, axis = 1)

# Creating a second class column.

passengers['SecondClass'] = passengers.apply(lambda row: 1 if (row['Pclass'] == 2) else 0, axis = 1)

# Alright! Now are data is cleaned and we use the columns above as features.

# Selecting the desired features.

cols = ['Sex', 'Age', 'FirstClass', 'SecondClass']
features = passengers[cols]
survival = passengers['Survived']

# Performing train, test, split.

X_train, X_test, y_train, y_test = train_test_split(features, survival)

# Scaling the feature data so it has mean = 0 and standard deviation = 1

standard_scaler = StandardScaler()
standard_scaler.fit_transform(X_train)
standard_scaler.transform(X_test)

# Creating and training the model.

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Scoring the model on the train data.

print(logreg.score(X_train, y_train))

# Scoring the model on the test data.

print(logreg.score(X_test, y_test))

# Analyzing the coefficients.

print(logreg.coef_)

# Sample passenger features for prediction

passenger_1 = np.array([0.0,20.0,0.0,0.0])
passenger_2 = np.array([1.0,17.0,1.0,0.0])

# Combining passenger arrays into one numpy array.

sample_passengers = np.array([passenger_1, passenger_2])

# Scaling the sample passenger features.

standard_scaler.transform(sample_passengers)

# Making survival predictions.

print(logreg.predict(sample_passengers))

print('\nThanks for reviewing')

# Thanks for reviewing

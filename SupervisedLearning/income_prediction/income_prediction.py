# In this program, we will try to predict whether or not a person makes more than $50,000.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# For more information about the dataset, please check:
# (https://archive.ics.uci.edu/ml/datasets/census%20income)

# Loading the data.

income_data = pd.read_csv('income.csv', header = 0, delimiter = ", ")
# print(income_data.head())

# We need to transform sex column into numerics.

income_data["sex_int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

cols = ["age", "sex_int", "capital-gain", "capital-loss", "hours-per-week"]
training = income_data[cols]
labels = income_data['income']

# Train, test, split

X_train, X_test, y_train, y_test = train_test_split(training, labels, random_state = 1)

# Creating and training a Random Forest model

clf = RandomForestClassifier(random_state = 1)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

print('\nThanks for reviewing')

# Thanks for reviewing

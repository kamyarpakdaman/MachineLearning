# In this program, we use Decision Trees to try to predict the continent of flags based on several features. We explore which features are the best to use and to create the tree model based on.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Note that:
# landmass: 1=N.America, 2=S.America, 3=Europe, 4=Africa, 4=Asia, 6=Oceania
# Hence, eventually, we are going to create a tree model to classify what landmass a country is on.

# Further information about other columns:
# (http://archive.ics.uci.edu/ml/datasets/Flags)

flags = pd.read_csv('flags.csv', header = 0)
# print(flags.head())

# Getting labels.

labels = flags['Landmass']

# We start by using only the colors of flag as a feature to predict the continent.

cols = ['Red',	'Green',	'Blue',	'Gold',	'White',	'Black',	'Orange']
data = flags[cols]

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 1)

# Creating and training the model.

clf = DecisionTreeClassifier(random_state = 1)
clf.fit(X_train, y_train)

# print(clf.score(X_train, y_train))
# print(clf.score(X_test, y_test))

# Alright. It is terrible. Let's move on. As follows, we tune the tree by increasing max_depth from 1 to 20.

max_depth_values = list(range(1, 21))
max_depth_accuracy_scores = []

for i in max_depth_values:
    clf = DecisionTreeClassifier(max_depth = i, random_state = 1)
    clf.fit(X_train, y_train)
    max_depth_accuracy_scores.append(clf.score(X_test, y_test))

plt.close('all')
plt.figure()

plt.plot(max_depth_values, max_depth_accuracy_scores)
plt.ylabel('Score')
plt.xlabel('Max depth')
plt.title('Score Change vs Max Depth Change\nwith Colors as features')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(max_depth_values)
ax.set_xticklabels(max_depth_values)
ax.tick_params(left = False, top = False, right = False, bottom = False)

plt.show()

# As the plot indicates, increasing depth will not help us much. Let's add more features now.

cols = ["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]
data = flags[cols]

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 1)

# Creating and training the model.

clf = DecisionTreeClassifier(random_state = 1)
clf.fit(X_train, y_train)

# print(clf.score(X_train, y_train))
# print(clf.score(X_test, y_test))

max_depth_values = list(range(1, 21))
max_depth_accuracy_scores = []

for i in max_depth_values:
    clf = DecisionTreeClassifier(max_depth = i, random_state = 1)
    clf.fit(X_train, y_train)
    max_depth_accuracy_scores.append(clf.score(X_test, y_test))

plt.close('all')
plt.figure()

plt.plot(max_depth_values, max_depth_accuracy_scores)
plt.ylabel('Score')
plt.xlabel('Max depth')
plt.title('Score Change vs Max Depth Change\nwith More features')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(max_depth_values)
ax.set_xticklabels(max_depth_values)
ax.tick_params(left = False, top = False, right = False, bottom = False)

plt.show()

# Here we see a clear improvement of around 20%.

print('\nThanks for reviewing')

# Thanks for reviewing

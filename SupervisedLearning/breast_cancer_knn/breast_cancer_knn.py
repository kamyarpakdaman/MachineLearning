# In this program, we create a k-NN classifier for breast cancer data and detect the best value for k.

from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

# Loading the data and having a look at its features and target values.

breast_cancer_data = load_breast_cancer()
# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)
# print(breast_cancer_data.target)
# print(breast_cancer_data.target_names)

# Splitting the data into train and test parts.

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

# Creating models and checking their performance for different values of k using a plot:

k_values = list(range(1, 101))
k_accuracies = []

for k in k_values:

    # Creating and training a k-NN classifier model.

    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)

    # Checking the accuracy of the model.

    score = clf.score(X_test, y_test)
    k_accuracies.append(score)

# Drawing a plot to show accuracy for k values plus distinguishing the maximum accuracy.

n_k_accuracies = np.array(k_accuracies)
max_index = np.argmax(n_k_accuracies)
# print(max_index)
k_max = k_values[max_index]
accuracy_max = k_accuracies[max_index]
# print(k_max, accuracy_max)

plt.figure()

plt.plot(k_values, k_accuracies, color = 'darkslateblue')
plt.vlines(k_max, 0.92, accuracy_max, color = 'springgreen', label = 'Max Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('K values')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left = False, bottom = False)
ax.text(k_max, accuracy_max+0.003, 'The best k is {}\nwith accuracy of {}%'.format(k_max, round((100*accuracy_max), 2)), color = 'black', ha = 'center')
plt.legend()

plt.show()

print('\nThanks for reviewing')

# Thanks for reviewing
